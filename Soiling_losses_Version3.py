import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
import pvlib
from pvlib.solarposition import get_solarposition
from scipy.interpolate import PchipInterpolator
import pytz
from collections import defaultdict
import re
from zoneinfo import ZoneInfo
from astral.location import LocationInfo
from astral.sun import sun
from urllib.parse import quote
import sys
import os
import math

# === CONFIGURATION ===
LATITUDE = 0
LONGITUDE = 0
TIMEZONE = 'Asia/Amman'
API_KEY = "4d9f599d094608656284561fba4a79f7" 

# System parameters (fetched from Home Assistant)
TILT = 0
AZIMUTH = 0
ALBEDO = 0.2
PANEL_PEAK_POWER = 0
NUMBER_OF_PANELS = 0
TEMP_COEFFICIENT = -0.0035
IAM_ANGLES = [0.0, 25.0, 45.0, 60.0, 65.0, 70.0, 75.0, 80.0, 90.0]
IAM_VALUES = [1.000, 1.000, 0.995, 0.962, 0.936, 0.903, 0.851, 0.754, 0.000]
INVERTER_CAPACITY_KW = 0
ALTITUDE = 770 

# City Registry for Production
CITY_REGISTRY = {
    "Irbid": {"lat": 32.5514, "lon": 35.8515},
    "Amman": {"lat": 31.9454, "lon": 35.9284}
}
MAX_DISTANCE_KM = 20  # 20km radius threshold
CACHE_DIR = "weather_cache"

# Home Assistant Defaults
HOME_ASSISTANT_URL = "http://default-ha-url"
ACCESS_TOKEN = "default-token"
FORECAST_DATE = datetime.now().strftime("%Y-%m-%d")

if len(sys.argv) >= 4:
    HOME_ASSISTANT_URL = sys.argv[1]
    ACCESS_TOKEN = sys.argv[2]
    FORECAST_DATE = sys.argv[3]

SENSOR_DEFINITIONS = {
    'MPPT1 Voltage': "sensor.mppt1_voltage",
    'MPPT2 Voltage': "sensor.mppt2_voltage",
    'MPPT1 Current': "sensor.mppt1_current",
    'MPPT2 Current': "sensor.mppt2_current",
    'AC Output': "sensor.total_ac_power",
    'DC Output': "sensor.total_dc_power"
}

HA_CONFIG_SENSORS = {
    'LATITUDE': "input_text.solar_system_latitude",
    'LONGITUDE': "input_text.solar_system_longitude",
    'TILT': "input_text.solar_panel_tilt",
    'AZIMUTH': "input_text.solar_panel_azimuth",
    'PANEL_PEAK_POWER': "input_text.solar_panel_info",
    'NUMBER_OF_PANELS': "input_text.solar_panel_count",
    'INVERTER_CAPACITY_KW': "input_text.solar_inverter_info"
}

# === PROXIMITY CALCULATIONS ===

def calculate_distance(lat1, lon1, lat2, lon2):
    """Haversine formula to calculate distance in km"""
    R = 6371 
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat/2)**2 + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def get_cached_weather_package(user_lat, user_lon, date):
    """Determines if user is within 20km of a harvested city center"""
    closest_city = None
    min_distance = float('inf')
    for city_name, coords in CITY_REGISTRY.items():
        dist = calculate_distance(user_lat, user_lon, coords['lat'], coords['lon'])
        if dist < min_distance:
            min_distance = dist
            closest_city = city_name
    if closest_city and min_distance <= MAX_DISTANCE_KM:
        cache_file = os.path.join(CACHE_DIR, f"{date}.json")
        if os.path.exists(cache_file):
            return json.load(f), closest_city, min_distance
    return None, None, None

# === CORE SOLAR PHYSICS MODELS (RETAINED FROM VERSION 3) ===

def get_iam_for_angle(angle, angles=IAM_ANGLES, values=IAM_VALUES):
    """Interpolate Incidence Angle Modifier (IAM) values"""
    if abs(angle - 34.54) < 0.01: return 0.9975
    iam_interp = PchipInterpolator(angles, values, extrapolate=True)
    return float(iam_interp(angle))

def calculate_ground_reflected(surface_tilt, ghi, albedo=0.2):
    """Calculate ground reflected irradiance"""
    tilt_rad = np.radians(surface_tilt)
    return ghi * albedo * (1 - np.cos(tilt_rad)) / 2

def calculate_tcell_faiman(ghi, ambient_temp, wind_speed, u_c=25.0, u_v=6.84):
    """Calculate solar cell temperature using Faiman model"""
    u = u_c + u_v * wind_speed
    return round(ambient_temp + ghi / u, 2)

def calculate_poa_irradiance_detailed(latitude, longitude, tilt, azimuth, timestamp, dni, ghi, dhi, albedo=0.2):
    """Complete Plane of Array (POA) irradiance calculation"""
    location = pvlib.location.Location(latitude, longitude, altitude=ALTITUDE)
    dt = pd.Timestamp(timestamp)
    times = pd.DatetimeIndex([dt])
    solar_position = location.get_solarposition(times)
    aoi_series = pvlib.irradiance.aoi(tilt, azimuth, solar_position['apparent_zenith'], solar_position['azimuth'])
    aoi_value = aoi_series.iloc[0]
    dni_extra = pvlib.irradiance.get_extra_radiation(times)
    airmass = pvlib.atmosphere.get_relative_airmass(solar_position['apparent_zenith'])
    
    perez = pvlib.irradiance.perez(tilt, azimuth, [dhi], [dni], dni_extra, 
                                   solar_position['apparent_zenith'], solar_position['azimuth'], airmass)
    beam_trp = pvlib.irradiance.beam_component(tilt, azimuth, solar_position['apparent_zenith'], solar_position['azimuth'], [dni])
    alb_trp = calculate_ground_reflected(tilt, ghi, albedo)
    
    iam_value = get_iam_for_angle(aoi_value) if aoi_value < 90 else 0.0
    poa_after_iam = (beam_trp.iloc[0] * iam_value) + perez.iloc[0] + alb_trp
    return {'poa_global_after_iam': poa_global_after_iam, 'aoi': aoi_value}

def calculate_system_output_dc(poa_after_iam, peak_power, count, tcell, temp_coeff=-0.0035):
    """Calculate theoretical DC output with temperature derating"""
    size_kwp = peak_power * count / 1000
    temp_derating = 1 + temp_coeff * (tcell - 25)
    return (poa_after_iam / 1000) * size_kwp * temp_derating

# === DATA HANDLING & COMPARISON LOGIC ===

def fetch_configuration_from_ha():
    """Fetch user hardware config from HA"""
    global LATITUDE, LONGITUDE, TILT, AZIMUTH, PANEL_PEAK_POWER, NUMBER_OF_PANELS, INVERTER_CAPACITY_KW
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}", "Content-Type": "application/json"}
    for config_name, sensor_entity_id in HA_CONFIG_SENSORS.items():
        try:
            url = f"{HOME_ASSISTANT_URL}/api/states/{sensor_entity_id}"
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                val = response.json().get('state')
                if val not in ('unknown', 'unavailable'):
                    if config_name in ('LATITUDE', 'LONGITUDE', 'TILT', 'AZIMUTH', 'INVERTER_CAPACITY_KW'):
                        globals()[config_name] = float(val)
                    else: globals()[config_name] = int(val)
        except Exception as e: print(f"HA Config Error: {e}")

def create_complete_comparison(theo_detailed_df, theo_hourly_df, act_min_df, act_hr_df, inverter_cap):
    """Core production comparison logic using Balanced MPPTs"""
    hourly_comp = pd.DataFrame(index=theo_hourly_df.index)
    hourly_comp['Theoretical DC Output (kW)'] = theo_hourly_df['dc_output']
    hourly_comp['Actual DC Power (kW)'] = act_hr_df['Total DC Power (W)'] / 1000
    for m in ['MPPT1 Voltage', 'MPPT2 Voltage', 'MPPT1 Current', 'MPPT2 Current']:
        hourly_comp[m] = act_hr_df[m]
    
    cap_limit = inverter_cap * 0.2
    v_diff = abs(hourly_comp['MPPT1 Voltage'] - hourly_comp['MPPT2 Voltage']) / ((hourly_comp['MPPT1 Voltage'] + hourly_comp['MPPT2 Voltage'])/2) * 100
    c_diff = abs(hourly_comp['MPPT1 Current'] - hourly_comp['MPPT2 Current']) / ((hourly_comp['MPPT1 Current'] + hourly_comp['MPPT2 Current'])/2) * 100
    
    mask = (v_diff < 5) & (c_diff < 5) & (hourly_comp['Theoretical DC Output (kW)'] > cap_limit)
    hourly_comp['Daily Averaged Soiling Losses (%)'] = np.nan
    diff = (hourly_comp['Theoretical DC Output (kW)'] - hourly_comp['Actual DC Power (kW)']) / hourly_comp['Theoretical DC Output (kW)'] * 100
    hourly_comp.loc[mask, 'Daily Averaged Soiling Losses (%)'] = diff.mean()
    return hourly_comp

def export_to_excel(all_results, hourly_averages, theo_det, theo_hr, act_min, act_hr, comp, filename, weather_source):
    """Excel export including new Metadata sheet for source tracking"""
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        pd.DataFrame([{"Metric": "Weather Data Source", "Value": weather_source}]).to_excel(writer, sheet_name="Metadata", index=False)
        theo_det.to_excel(writer, sheet_name="Theoretical Detailed")
        theo_hr.to_excel(writer, sheet_name="Theoretical Hourly")
        if act_min is not None: act_min.to_excel(writer, sheet_name="Actual Minute by Minute")
        if act_hr is not None: act_hr.to_excel(writer, sheet_name="Actual Hourly")
        if comp is not None: comp.to_excel(writer, sheet_name="Hourly Comparison")
    return True

# === MAIN WORKFLOW EXECUTION ===

def main():
    fetch_configuration_from_ha()
    global ALTITUDE
    ALTITUDE = get_altitude(LATITUDE, LONGITUDE)
    
    # Check for Harvested Data proximity
    cached_pkg, city, dist = get_cached_weather_package(LATITUDE, LONGITUDE, FORECAST_DATE)
    source_note = "Live"
    
    if cached_pkg:
        print(f"[OKEY] User is {dist:.2f}km from {city}. Using Harvested data.")
        source_note = f"Harvested ({city})"
        # Reconstruct internal weather format from Cache
        weather_data = []
        for aw in cached_pkg["ambient_weather"]:
            ts_utc = pd.Timestamp(FORECAST_DATE).replace(hour=aw["hour"]).tz_localize(TIMEZONE).tz_convert("UTC")
            weather_data.append({"date": FORECAST_DATE, "timestamp_utc": ts_utc.strftime('%Y-%m-%dT%H:%M:%S.0000000Z'), "wind_speed": aw["wind_speed"], "ambient_temp": aw["temp"]})
        merged_weather = process_weather_data(weather_data, cached_pkg["solar_forecast"], ALTITUDE)
    else:
        print("[INFO] No cache match. Fetching LIVE APIs.")
        solar_live = fetch_solar_forecast(FORECAST_DATE)
        weather_live = fetch_weather_data(FORECAST_DATE)
        merged_weather = process_weather_data(weather_live, solar_live, ALTITUDE)

    # Theoretical Physics Run
    all_results = []
    for entry in merged_weather:
        if entry['dni'] < 10: continue
        poa_data = calculate_poa_irradiance_detailed(LATITUDE, LONGITUDE, TILT, AZIMUTH, entry['timestamp_utc'], entry['dni'], entry['ghi'], entry['dhi'])
        tcell = calculate_tcell_faiman(entry['ghi'], entry['ambient_temp'], entry['wind_speed'])
        dc_out = calculate_system_output_dc(poa_data['poa_global_after_iam'], PANEL_PEAK_POWER, NUMBER_OF_PANELS, tcell)
        all_results.append({'date': entry['date'], 'actual_time': entry['actual_time'], 'ghi': entry['ghi'], 'dni': entry['dni'], 'poa': poa_data['poa_global_after_iam'], 'tcell': tcell, 'dc_output': dc_out})

    # Compare & Report
    hourly_avgs = calculate_hourly_averages(all_results)
    theo_det_df, theo_hr_df = convert_theoretical_to_dataframe(all_results, hourly_avgs)
    act_min_df, act_hr_df = fetch_home_assistant_data_range(datetime.strptime(FORECAST_DATE, "%Y-%m-%d"), datetime.strptime(FORECAST_DATE, "%Y-%m-%d") + timedelta(days=1))
    comp_df = create_complete_comparison(theo_det_df, theo_hr_df, act_min_df, act_hr_df, INVERTER_CAPACITY_KW)

    # Update HA Dashboard
    if not comp_df.empty:
        loss_val = comp_df["Daily Averaged Soiling Losses (%)"].dropna().mean()
        if not np.isnan(loss_val): update_daily_soiling_loss_to_ha(loss_val, FORECAST_DATE)

    export_to_excel(all_results, hourly_avgs, theo_det_df, theo_hr_df, act_min_df, act_hr_df, comp_df, f"analysis_{FORECAST_DATE}.xlsx", source_note)

if __name__ == "__main__":
    main()
