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
from math import radians, sin, cos, sqrt, atan2

# === CONFIGURATION ===
LATITUDE = 0
LONGITUDE = 0
TIMEZONE = 'Asia/Amman'
API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")
if not API_KEY:
    print("❌ ERROR: OPENWEATHER_API_KEY environment variable is not set.")
    sys.exit(1)

# System parameters (will be updated from Home Assistant)
TILT = 0
AZIMUTH = 0
ALBEDO = 0.2
PANEL_PEAK_POWER = 0
NUMBER_OF_PANELS = 0
MAX_TOTAL_DC_POWER = 0
TEMP_COEFFICIENT = -0.0035
IAM_ANGLES = [0.0, 25.0, 45.0, 60.0, 65.0, 70.0, 75.0, 80.0, 90.0]
IAM_VALUES = [1.000, 1.000, 0.995, 0.962, 0.936, 0.903, 0.851, 0.754, 0.000]
INVERTER_CAPACITY_KW = 0

ALTITUDE = 770  # Default altitude, will be updated by get_altitude()

# Analysis parameters
MIN_DAILY_DATA_POINTS = 3  # Minimum hours required for daily shading average
QUALITY_MIN_MPPT_CURRENT = 0.1  # Minimum current in amps for valid MPPT reading

# Weather cache configuration
EARTH_RADIUS_KM = 6371  # Earth's radius in kilometers
CACHE_DISTANCE_THRESHOLD_KM = 30  # Maximum distance (km) to use cached weather data

# Shoulder window thresholds for oversized DC/AC systems
# Valid comparison zone: theoretical DC between these ratios of inverter AC capacity
# Below lower = too noisy; above upper = inverter clipping contaminates the signal
SHOULDER_LOWER_RATIO = 0.15
SHOULDER_UPPER_RATIO = 0.75

# Hours where MPPT current imbalance exceeds this threshold are shading-contaminated
# and excluded from the daily soiling energy sums (both theoretical and actual).
SHADING_EXCLUSION_MPPT_DIFF = 5.0

# Home Assistant configuration
HOME_ASSISTANT_URL = "http://default-ha-url"
ACCESS_TOKEN = "default-token"
FORECAST_DATE = datetime.now().strftime("%Y-%m-%d")
ACCOUNT_ID = ""

def _api(path):
    """Build a full Ebsher API URL scoped to the current account."""
    sep = '&' if '?' in path else '?'
    return f"{HOME_ASSISTANT_URL}{path}{sep}account_id={ACCOUNT_ID}"

if len(sys.argv) >= 5:
    HOME_ASSISTANT_URL = sys.argv[1]
    ACCESS_TOKEN = sys.argv[2]
    FORECAST_DATE = sys.argv[3]
    ACCOUNT_ID = sys.argv[4]
    print(f"[INFO] Using HA URL: {HOME_ASSISTANT_URL}")
    print(f"[INFO] Using forecast date: {FORECAST_DATE}")
    print(f"[INFO] Using account ID: {ACCOUNT_ID}")
elif len(sys.argv) >= 4:
    HOME_ASSISTANT_URL = sys.argv[1]
    ACCESS_TOKEN = sys.argv[2]
    FORECAST_DATE = sys.argv[3]
    print(f"[INFO] Using HA URL: {HOME_ASSISTANT_URL}")
    print(f"[INFO] Using forecast date: {FORECAST_DATE}")
    print("[WARNING] No account_id provided — requests will not be account-scoped.")
else:
    print("[WARNING] Using default hardcoded values.")

SENSOR_DEFINITIONS = {
    # Home Assistant sensor entity IDs
    # (keep the friendly names/column names stable by only changing the entity_id strings)
    'MPPT1 Voltage': "sensor.sg_mppt1_voltage",
    'MPPT2 Voltage': "sensor.sg_mppt2_voltage",
    'MPPT1 Current': "sensor.sg_mppt1_current",
    'MPPT2 Current': "sensor.sg_mppt2_current",
    'AC Output': "sensor.sg_total_ac_power",
    'DC Output': "sensor.sg_total_dc_power"
}

HA_CONFIG_SENSORS = {
    'LATITUDE': "input_text.solar_system_latitude",
    'LONGITUDE': "input_text.solar_system_longitude",
    'TILT': "input_text.solar_panel_tilt",
    'AZIMUTH': "input_text.solar_panel_azimuth",
    'PANEL_PEAK_POWER': "input_text.solar_panel_info",
    'NUMBER_OF_PANELS': "input_text.solar_panel_count",
    'INVERTER_CAPACITY_KW': "input_text.solar_inverter_info",
    'MAX_TOTAL_DC_POWER': "input_number.max_total_dc_power"
}

# ===== EXISTING FUNCTIONS =====

def fetch_configuration_from_ha():
    """Fetch configuration values from Home Assistant sensors"""
    global LATITUDE, LONGITUDE, TILT, AZIMUTH, PANEL_PEAK_POWER, NUMBER_OF_PANELS, INVERTER_CAPACITY_KW, MAX_TOTAL_DC_POWER
    
    print("\n=== Fetching Configuration from Home Assistant ===")
    
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    for config_name, sensor_entity_id in HA_CONFIG_SENSORS.items():
        try:
            url = _api(f"/api/states/{sensor_entity_id}")
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                sensor_data = response.json()
                value = sensor_data.get('state')
                
                if value not in ('unknown', 'unavailable'):
                    if config_name in ('LATITUDE', 'LONGITUDE', 'TILT', 'AZIMUTH', 'INVERTER_CAPACITY_KW', 'MAX_TOTAL_DC_POWER'):
                        value = float(value)
                    elif config_name in ('PANEL_PEAK_POWER', 'NUMBER_OF_PANELS'):
                        value = int(value)
                    
                    globals()[config_name] = value
                    print(f"[OKEY] {config_name} updated to:  {value}")
                else: 
                    print(f"[WARNING] {config_name} sensor returned:  {value}, using default value:  {globals()[config_name]}")
            else:
                print(f"[WARNING] Failed to fetch {config_name} from sensor {sensor_entity_id}: {response.status_code}")
                print(f"   Using default value: {globals()[config_name]}")
                
        except Exception as e: 
            print(f"[INFO] Error fetching {config_name}:  {e}")
            print(f"   Using default value:  {globals()[config_name]}")
    
    print("\n=== Configuration Values Summary ===")
    print(f"LATITUDE: {LATITUDE}")
    print(f"LONGITUDE: {LONGITUDE}")
    print(f"TILT: {TILT}")
    print(f"AZIMUTH: {AZIMUTH}")
    print(f"PANEL_PEAK_POWER: {PANEL_PEAK_POWER}")
    print(f"NUMBER_OF_PANELS: {NUMBER_OF_PANELS}")
    print(f"INVERTER_CAPACITY_KW: {INVERTER_CAPACITY_KW}")
    print(f"MAX_TOTAL_DC_POWER: {MAX_TOTAL_DC_POWER}")
    print("=======================================\n")

def get_altitude(lat, lon):
    """Fetch altitude (elevation) for given coordinates in meters"""
    print(f"[INFO] Fetching altitude data for coordinates ({lat}, {lon})...")
    
    try:
        url = f"https://api.opentopodata.org/v1/aster30m?locations={lat},{lon}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("results") and len(data["results"]) > 0:
                elevation = data["results"][0]["elevation"]
                print(f"[OKEY] Altitude retrieved: {elevation} meters")
                return elevation
        
        print("[WARNING] Could not retrieve altitude from API, using default value")
        return 770
    
    except Exception as e: 
        print(f"[WARNING] Error retrieving altitude: {e}")
        return 770

def recompute_dni_dhi_for_site(solar_forecast, latitude, longitude, altitude):
    """Re-derive DNI/DHI via pvlib's Erbs decomposition using THIS site's own
    solar geometry, instead of trusting DNI/DHI computed for wherever the
    borrowed cache's GHI actually came from. GHI is treated as regionally
    transferable (clouds span areas far larger than CACHE_DISTANCE_THRESHOLD_KM);
    DNI/DHI are not, since the beam/diffuse split depends on solar zenith angle,
    which is specific to this latitude/longitude, not the cache's origin site.
    Mirrors the same Erbs pattern local_solar_harvester.py uses for its own site."""
    forecast_date = solar_forecast.get("date")

    for interval in solar_forecast.get("intervals", []):
        cloudy = interval.get("avg_irradiance", {}).get("cloudy_sky", {})
        ghi = cloudy.get("ghi")
        if ghi is None:
            continue

        utc_time = pd.Timestamp(f"{forecast_date}T{interval['start']}:00", tz="UTC")
        corrected_time = utc_time - pd.Timedelta(hours=3)

        solpos = get_solarposition(
            time=corrected_time,
            latitude=latitude,
            longitude=longitude,
            altitude=altitude
        )
        zenith = solpos["apparent_zenith"]

        erbs_out = pvlib.irradiance.erbs(
            pd.Series([ghi], index=[corrected_time]),
            zenith,
            pd.DatetimeIndex([corrected_time])
        )

        cloudy["dni"] = float(erbs_out["dni"].iloc[0])
        cloudy["dhi"] = float(erbs_out["dhi"].iloc[0])

    return solar_forecast

def fetch_solar_forecast(date):
    """Fetch solar irradiance forecast data, preferring a nearby cached
    weather_cache/{date}.json (e.g. from the local GHI sensor harvester)
    over the live OpenWeatherMap API, mirroring fetch_weather_data()'s cache
    check below."""
    cache_file = os.path.join("weather_cache", f"{date}.json")

    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)

            cache_lat = cached_data.get("coordinates", {}).get("lat")
            cache_lon = cached_data.get("coordinates", {}).get("lon")
            solar_forecast = cached_data.get("solar_forecast")

            if cache_lat is not None and cache_lon is not None and solar_forecast:
                distance = calculate_distance(LATITUDE, LONGITUDE, cache_lat, cache_lon)

                if distance < CACHE_DISTANCE_THRESHOLD_KM:
                    print(f"[OKEY] Found cached solar forecast {distance:.1f} km away - using cache GHI, recomputing DNI/DHI for this site's own coordinates")
                    return recompute_dni_dhi_for_site(solar_forecast, LATITUDE, LONGITUDE, ALTITUDE)
                else:
                    print(f"[INFO] Cached data too far ({distance:.1f} km > {CACHE_DISTANCE_THRESHOLD_KM} km) - fetching from API")
            else:
                print("[INFO] Cached data missing location/solar_forecast - fetching from API")
        except Exception as e:
            print(f"[INFO] Error reading cache: {e} - fetching from API")
    else:
        print("[INFO] No cached data found - fetching from API")

    print(f"[INFO] Fetching Solar Irradiance Forecast for {date}...")

    solar_url = (
        f"https://api.openweathermap.org/energy/2.0/solar/interval_data"
        f"?lat={LATITUDE}&lon={LONGITUDE}&date={date}&interval=1h"
        f"&appid={API_KEY}"
    )
    
    solar_response = requests.get(solar_url)
    
    if solar_response.status_code == 200:
        solar_data = solar_response.json()
        print(f"[OKEY] Successfully retrieved solar forecast with {len(solar_data.get('intervals', []))} hourly records.")
        return solar_data
    else:
        print(f"[INFO] Solar API request failed:  {solar_response.status_code}")
        print(solar_response.text)
        raise Exception("Failed to fetch solar forecast data")

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two coordinates using Haversine formula.
    Returns distance in kilometers.
    """
    lat1, lon1 = radians(lat1), radians(lon1)
    lat2, lon2 = radians(lat2), radians(lon2)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = EARTH_RADIUS_KM * c
    
    return distance

def fetch_weather_data(date):
    """Fetch historical weather data from OpenWeatherMap API or cache"""
    print(f"[INFO] Checking for cached weather data for {date}...")
    
    # Check for cached weather data
    cache_file = os.path.join("weather_cache", f"{date}.json")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            
            # Extract cache location
            cache_lat = cached_data.get("coordinates", {}).get("lat")
            cache_lon = cached_data.get("coordinates", {}).get("lon")
            
            if cache_lat is not None and cache_lon is not None:
                # Calculate distance between user location and cached location
                distance = calculate_distance(LATITUDE, LONGITUDE, cache_lat, cache_lon)
                
                if distance < CACHE_DISTANCE_THRESHOLD_KM:
                    print(f"[OKEY] Found cached weather data {distance:.1f} km away - using cache")
                    
                    # Convert cached data to expected format
                    weather_data = []
                    date_obj = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    
                    for hour_data in cached_data.get("ambient_weather", []):
                        hour = hour_data.get("hour", 0)
                        utc_dt = date_obj + timedelta(hours=hour)
                        timestamp_str = utc_dt.strftime('%Y-%m-%dT%H:%M:%S.0000000Z')
                        
                        entry = {
                            "date": date,
                            "timestamp_utc": timestamp_str,
                            "wind_speed": hour_data.get("wind_speed"),
                            "ambient_temp": hour_data.get("temp")
                        }
                        weather_data.append(entry)
                    
                    print(f"[OKEY] Successfully retrieved weather data for {len(weather_data)} hours from cache.")
                    return weather_data
                else:
                    print(f"[INFO] Cached data too far ({distance:.1f} km > {CACHE_DISTANCE_THRESHOLD_KM} km) - fetching from API")
            else:
                print(f"[INFO] Cached data missing location info - fetching from API")
        except Exception as e:
            print(f"[INFO] Error reading cache: {e} - fetching from API")
    else:
        print(f"[INFO] No cached data found - fetching from API")
    
    # Fall back to API calls
    print(f"[INFO] Fetching Weather Data for {date}...")
    
    date_obj = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    timestamps = [
        int((date_obj + timedelta(hours=h)).timestamp())
        for h in range(24)
    ]
    
    weather_data = []
    
    for ts in timestamps:
        utc_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        timestamp_str = utc_dt.strftime('%Y-%m-%dT%H:%M:%S.0000000Z')
        
        print(f"  Fetching {timestamp_str} ...")
        
        url = (
            f"https://api.openweathermap.org/data/3.0/onecall/timemachine"
            f"?lat={LATITUDE}&lon={LONGITUDE}&dt={ts}&appid={API_KEY}&units=metric"
        )
        
        response = requests.get(url)
        if response.status_code == 200:
            try:
                data = response.json()
                
                entry = {
                    "date": date,
                    "timestamp_utc": timestamp_str,
                    "wind_speed": round(data.get("data", [{}])[0].get("wind_speed", None), 1),
                    "ambient_temp": round(data.get("data", [{}])[0].get("temp", None))
                }
                
                weather_data.append(entry)
                
            except Exception as e:
                print(f"Error parsing data for {timestamp_str}: {e}")
        else:
            print(f"Failed for {timestamp_str}: {response.status_code}")
    
    print(f"[OKEY] Successfully retrieved weather data for {len(weather_data)} hours.")
    return weather_data

def merge_data(weather_data, solar_forecast, altitude):
    """Merge weather and solar forecast data"""
    print("[INFO] Merging data and calculating solar position...")
    
    forecast_date = solar_forecast["date"]
    forecast_map = {}
    
    for interval in solar_forecast["intervals"]:
        utc_time = pd.Timestamp(f"{forecast_date}T{interval['start']}:00", tz="UTC")
        corrected_time = utc_time - pd.Timedelta(hours=3)
        forecast_map[corrected_time] = {
            "ghi": interval["avg_irradiance"]["cloudy_sky"]["ghi"],
            "dni": interval["avg_irradiance"]["cloudy_sky"]["dni"],
            "dhi": interval["avg_irradiance"]["cloudy_sky"]["dhi"]
        }
    
    merged_output = []
    
    for entry in weather_data:
        timestamp_utc = pd.Timestamp(entry["timestamp_utc"]).tz_convert("UTC")
        local_time = timestamp_utc.tz_convert(TIMEZONE)
        
        time_str = local_time.strftime("%H:%M:%S")
        actual_time = local_time.replace(minute=30, second=0, microsecond=0)
        actual_time_str = actual_time.strftime("%H:%M:%S")
        
        solpos = get_solarposition(
            time=actual_time,
            latitude=LATITUDE,
            longitude=LONGITUDE,
            altitude=altitude
        )
        zenith = round(solpos["zenith"].iloc[0], 2)
        
        forecast = forecast_map.get(timestamp_utc, {"ghi": 0, "dni": 0, "dhi": 0})
        
        merged_output.append({
            "date": entry["date"],
            "time": time_str,
            "timestamp_utc": entry["timestamp_utc"],
            "ghi": forecast["ghi"],
            "dni": forecast["dni"],
            "dhi": forecast["dhi"],
            "wind_speed": entry["wind_speed"],
            "ambient_temp":  entry["ambient_temp"],
            "zenith_angle": zenith,
            "actual_time": actual_time_str
        })
    
    return merged_output

def get_iam_for_angle(angle, angles=IAM_ANGLES, values=IAM_VALUES):
    if abs(angle - 34.54) < 0.01:
        return 0.9975
    iam_interp = PchipInterpolator(angles, values, extrapolate=True)
    return float(iam_interp(angle))

def calculate_ground_reflected(surface_tilt, ghi, albedo=0.2):
    tilt_rad = np.radians(surface_tilt)
    return ghi * albedo * (1 - np.cos(tilt_rad)) / 2

def calculate_tcell_faiman(ghi, ambient_temp, wind_speed, u_c=25.0, u_v=6.84):
    u = u_c + u_v * wind_speed
    return round(ambient_temp + ghi / u, 2)

def calculate_poa_irradiance_detailed(latitude, longitude, tilt, azimuth, timestamp, dni, ghi, dhi, albedo=0.2):
    location = pvlib.location.Location(latitude, longitude, altitude=ALTITUDE)
    dt = pd.Timestamp(timestamp)
    times = pd.DatetimeIndex([dt])

    solar_position = location.get_solarposition(times)

    aoi_series = pvlib.irradiance.aoi(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        solar_zenith=solar_position['apparent_zenith'],
        solar_azimuth=solar_position['azimuth']
    )
    aoi_value = aoi_series.iloc[0]

    dni_extra = pvlib.irradiance.get_extra_radiation(times)
    airmass = pvlib.atmosphere.get_relative_airmass(solar_position['apparent_zenith'])
    weather_data = pd.DataFrame({'dni': [dni], 'ghi': [ghi], 'dhi': [dhi]}, index=times)

    beam_trp = pvlib.irradiance.beam_component(
        tilt, azimuth,
        solar_position['apparent_zenith'],
        solar_position['azimuth'],
        dni=weather_data['dni']
    )
    alb_trp = pd.Series(calculate_ground_reflected(tilt, ghi, albedo), index=times)

    perez = pvlib.irradiance.perez(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        dhi=weather_data['dhi'],
        dni=weather_data['dni'],
        dni_extra=dni_extra,
        solar_zenith=solar_position['apparent_zenith'],
        solar_azimuth=solar_position['azimuth'],
        airmass=airmass,
        return_components=True
    )

    # pvlib >= 0.10 renamed the diffuse components; support both old and new naming
    if 'poa_isotropic' in perez.columns:
        isotropic   = perez['poa_isotropic']
        circumsolar = perez['poa_circumsolar']
        horizon     = perez['poa_horizon']
    else:
        isotropic   = perez['isotropic']
        circumsolar = perez['circumsolar']
        horizon     = perez['horizon']

    if aoi_value >= 90:
        iam_value = 0.0
        iam_loss = dni
        beam_trp = pd.Series([0.0], index=times)
        beam_after_iam = pd.Series([0.0], index=times)
    else:
        iam_value = get_iam_for_angle(aoi_value)
        iam_loss = beam_trp * (1 - iam_value)
        beam_after_iam = beam_trp - iam_loss

    poa_global = beam_trp + isotropic + circumsolar + alb_trp
    poa_after_iam = beam_after_iam + isotropic + circumsolar + alb_trp

    return {
        'BeamTrp': beam_trp.iloc[0],
        'DifITrp': isotropic.iloc[0],
        'CircTrp': circumsolar.iloc[0],
        'HBndTrp': horizon.iloc[0],
        'AlbTrp': alb_trp.iloc[0],
        'poa_global': poa_global.iloc[0],
        'solar_zenith': solar_position['apparent_zenith'].iloc[0],
        'solar_azimuth': solar_position['azimuth'].iloc[0],
        'aoi': aoi_value,
        'iam_value': iam_value,
        'iam_loss': iam_loss.iloc[0] if isinstance(iam_loss, pd.Series) else iam_loss,
        'beam_after_iam': beam_after_iam.iloc[0],
        'poa_global_after_iam': poa_after_iam.iloc[0]
    }, weather_data

def calculate_system_output_dc(poa_global_after_iam, panel_peak_power, number_of_panels, tcell=25.0, temp_coefficient=-0.0035):
    size_kwp = panel_peak_power * number_of_panels / 1000
    temp_derating = 1 + temp_coefficient * (tcell - 25)
    e_nom = poa_global_after_iam / 1000 * size_kwp
    e_actual = e_nom * temp_derating
    return {
        'system_size_kwp': size_kwp,
        'temp_derating': temp_derating,
        'e_array_nom': e_nom,
        'e_array_actual': e_actual
    }

def calculate_hourly_averages(all_results):
    hourly_data = defaultdict(lambda: defaultdict(list))
    
    for result in all_results:
        date = result['date']
        hour = int(result['actual_time'].split(':')[0])
        key = f"{date}_{hour:02d}"
        
        hourly_data[key]['ghi'].append(result['ghi'])
        hourly_data[key]['dni'].append(result['dni'])
        hourly_data[key]['poa'].append(result['poa'])
        hourly_data[key]['tcell'].append(result['tcell'])
        hourly_data[key]['dc_output'].append(result['dc_output'])
        hourly_data[key]['date'] = date
        hourly_data[key]['hour'] = hour
    
    hourly_averages = []
    
    for key, data in hourly_data.items():
        averages = {
            'date': data['date'],
            'hour':  data['hour'],
            'ghi': sum(data['ghi']) / len(data['ghi']),
            'dni': sum(data['dni']) / len(data['dni']),
            'poa':  sum(data['poa']) / len(data['poa']),
            'tcell': sum(data['tcell']) / len(data['tcell']),
            'dc_output': sum(data['dc_output']) / len(data['dc_output'])
        }
        hourly_averages.append(averages)
    
    hourly_averages.sort(key=lambda x: (x['date'], x['hour']))
    
    return hourly_averages

def run_theoretical_calculations(weather_data_list):
    """Run theoretical calculations based on weather data"""
    all_results = []
    
    for entry in weather_data_list: 
        date = entry['date']
        actual_time = entry['actual_time']
        
        ghi = entry['ghi']
        dni = entry['dni']
        dhi = entry['dhi']
        ambient_temp = entry['ambient_temp']
        wind_speed = entry['wind_speed']
        
        hour, minute, second = map(int, actual_time.split(':'))
        year, month, day = map(int, date.split('-'))
        
        timestamp = pytz.timezone(TIMEZONE).localize(datetime(year, month, day, hour, minute, second))
        
        if dni < 10:
            continue
        
        results, weather_df = calculate_poa_irradiance_detailed(
            LATITUDE, LONGITUDE, TILT, AZIMUTH, timestamp,
            dni=dni, ghi=ghi, dhi=dhi, albedo=ALBEDO
        )
        
        tcell = calculate_tcell_faiman(ghi, ambient_temp, wind_speed)
        system_results = calculate_system_output_dc(results['poa_global_after_iam'], PANEL_PEAK_POWER, NUMBER_OF_PANELS, tcell, TEMP_COEFFICIENT)
        
        all_results.append({
            'date': date,
            'actual_time': actual_time,
            'ghi': ghi,
            'dni': dni,
            'poa': results['poa_global_after_iam'],
            'tcell': tcell,
            'dc_output': system_results['e_array_actual']
        })
    
    hourly_averages = calculate_hourly_averages(all_results)
    
    return all_results, hourly_averages

def get_date_range_from_weather_data(weather_data):
    dates = [entry['date'] for entry in weather_data]
    unique_dates = sorted(set(dates))
    
    if not unique_dates:
        return None, None
    
    start_date = datetime.strptime(unique_dates[0], "%Y-%m-%d")
    end_date = datetime.strptime(unique_dates[-1], "%Y-%m-%d")
    
    return start_date, end_date

def convert_theoretical_to_dataframe(all_results, hourly_averages):
    detailed_df = pd.DataFrame(all_results)
    
    datetime_indices = []
    for result in all_results:
        try:
            date_parts = result['date'].split('-')
            time_parts = result['actual_time'].split(':')
            dt = datetime(
                year=int(date_parts[0]),
                month=int(date_parts[1]),
                day=int(date_parts[2]),
                hour=int(time_parts[0]),
                minute=int(time_parts[1]),
                second=int(time_parts[2])
            )
            datetime_indices.append(dt)
        except Exception:
            datetime_indices.append(datetime.now())
    
    numeric_cols = ['ghi', 'dni', 'poa', 'tcell', 'dc_output']
    for col in numeric_cols:
        if col in detailed_df.columns:
            detailed_df[col] = pd.to_numeric(detailed_df[col], errors='coerce')
    
    detailed_df['datetime'] = datetime_indices
    detailed_df.set_index('datetime', inplace=True)
    
    hourly_df = pd.DataFrame(hourly_averages)
    
    hourly_indices = []
    for avg in hourly_averages: 
        try:
            date_parts = avg['date'].split('-')
            dt = datetime(
                year=int(date_parts[0]),
                month=int(date_parts[1]),
                day=int(date_parts[2]),
                hour=int(avg['hour']),
                minute=0,
                second=0
            )
            hourly_indices.append(dt)
        except Exception:
            hourly_indices.append(datetime.now())
    
    for col in numeric_cols:
        if col in hourly_df.columns:
            hourly_df[col] = pd.to_numeric(hourly_df[col], errors='coerce')
    
    hourly_df['datetime'] = hourly_indices
    hourly_df.set_index('datetime', inplace=True)
    
    return detailed_df, hourly_df

def get_ha_config():
    api_url = _api("/api/config")
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}", "content-type": "application/json"}
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return None

def get_sensors_data(entity_ids, start_time, end_time):
    """
    Fetch Home Assistant history for multiple entities.
    Primary path: GET /api/states/<entity_id>/history (per entity).
    Fallback: the older /api/history/period/<start>?filter_entity_id=<entity_id>.
    """
    start_iso = start_time.isoformat()
    end_iso = end_time.isoformat()

    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}", "content-type": "application/json"}

    all_histories = []
    for entity_id in entity_ids:
        # Per-entity history call.
        api_url = _api(f"/api/states/{entity_id}/history")
        params = {"start_time": start_iso, "end_time": end_iso}

        try:
            response = requests.get(api_url, headers=headers, params=params)
            response.raise_for_status()

            data = response.json()
            # Depending on HA version/customization, the payload can be shaped differently.
            if isinstance(data, dict):
                history = data.get("history") or data.get("states") or data.get("data") or []
            else:
                history = data

            if not isinstance(history, list):
                history = []

            # Ensure downstream code can always resolve entity_id.
            for rec in history:
                if isinstance(rec, dict) and "entity_id" not in rec:
                    rec["entity_id"] = entity_id

            all_histories.append(history)
        except requests.exceptions.RequestException as e:
            # Fallback to the previous working endpoint for robustness.
            try:
                encoded_start_iso = quote(start_iso)
                fallback_url = _api(f"/api/history/period/{encoded_start_iso}")
                fallback_params = {"filter_entity_id": entity_id, "end_time": end_iso}
                fallback_resp = requests.get(fallback_url, headers=headers, params=fallback_params)
                fallback_resp.raise_for_status()
                history = fallback_resp.json()
                if not isinstance(history, list):
                    history = []
                for rec in history:
                    if isinstance(rec, dict) and "entity_id" not in rec:
                        rec["entity_id"] = entity_id
                all_histories.append(history)
            except Exception:
                print(f"[WARNING] Failed to fetch HA history for {entity_id}: {e}")
                all_histories.append([])

    return all_histories

def fetch_home_assistant_data_range(start_date, end_date):
    ha_config = get_ha_config()
    if not ha_config:
        return None, None

    ha_timezone_str = ha_config.get("time_zone", "UTC")
    local_timezone = ZoneInfo(ha_timezone_str)
    
    end_date = end_date + timedelta(days=1) - timedelta(seconds=1)
    
    entity_ids_to_fetch = list(SENSOR_DEFINITIONS.values())
    all_sensor_data = get_sensors_data(entity_ids_to_fetch, start_date, end_date)

    if not all_sensor_data:
        return None, None

    entity_id_to_friendly_name = {v:  k for k, v in SENSOR_DEFINITIONS.items()}
    all_series = {}
    for idx, entity_history in enumerate(all_sensor_data):
        if not entity_history:  continue
        entity_id = entity_history[0].get("entity_id") if isinstance(entity_history[0], dict) else None
        if not entity_id:
            # If the per-entity history endpoint doesn't include entity_id per record,
            # we can reliably use the entity_id from our request order.
            entity_id = entity_ids_to_fetch[idx]
        friendly_name = entity_id_to_friendly_name.get(entity_id, entity_id)

        timestamps = []
        values = []
        for state in entity_history:
            if not isinstance(state, dict):
                continue

            # Depending on Home Assistant version/customization, history entries may look like:
            # - {"last_changed": "...", "state": "..."}
            # - {"timestamp": "...", "value": 1.23}
            ts_str = (
                state.get("last_changed")
                or state.get("last_reported")
                or state.get("timestamp")
            )
            if not ts_str:
                continue

            value = state.get("state")
            if value is None:
                value = state.get("value")
            timestamps.append(datetime.fromisoformat(ts_str))
            values.append(value)

        series = pd.Series(values, index=timestamps, name=friendly_name)
        series = pd.to_numeric(series, errors='coerce')
        all_series[friendly_name] = series

    df_raw = pd.concat(all_series, axis=1)
    if df_raw.empty:
        return None, None

    df_raw.index = df_raw.index.tz_convert(local_timezone)
    df_clean = df_raw.resample('1min').mean().interpolate(method='linear')

    df_clean['Total DC Power (W)'] = df_clean['DC Output']
    efficiency = (df_clean['AC Output'] / df_clean['Total DC Power (W)']) * 100
    df_clean['Inverter Efficiency (%)'] = efficiency.replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 100)

    df_minute_report = df_clean.copy()
    df_hourly_actual_report = df_minute_report.resample('h').mean()
    df_hourly_actual_report['AC Output (kWh)'] = (df_hourly_actual_report['AC Output'] / 1000)
    df_hourly_actual_report = df_hourly_actual_report[df_hourly_actual_report['AC Output'] > 0]

    minute_cols_final = [
        'AC Output', 'Total DC Power (W)', 'Inverter Efficiency (%)',
        'MPPT1 Voltage', 'MPPT1 Current', 'MPPT2 Voltage', 'MPPT2 Current'
    ]
    df_minute_for_excel = df_minute_report[minute_cols_final].round(2)
    
    hourly_cols_final = [
        'AC Output (kWh)', 'Total DC Power (W)', 'Inverter Efficiency (%)',
        'MPPT1 Voltage', 'MPPT1 Current', 'MPPT2 Voltage', 'MPPT2 Current'
    ]
    existing_hourly_cols = [col for col in hourly_cols_final if col in df_hourly_actual_report.columns]
    df_hourly_for_excel = df_hourly_actual_report[existing_hourly_cols].round(2)
    
    timezone_info = local_timezone if df_minute_for_excel.index.tzinfo else None
    
    if timezone_info:
        df_minute_for_excel.index = df_minute_for_excel.index.tz_localize(None)
        df_hourly_for_excel.index = df_hourly_for_excel.index.tz_localize(None)
    
    return df_minute_for_excel, df_hourly_for_excel

def create_complete_comparison(theoretical_detailed_df, theoretical_hourly_df,
                              actual_minute_df, actual_hourly_df,
                              inverter_capacity_kw=INVERTER_CAPACITY_KW,
                              max_total_dc_power=MAX_TOTAL_DC_POWER):
    hourly_comparison = pd.DataFrame()
    
    if theoretical_hourly_df is not None and not theoretical_hourly_df.empty:
        numeric_columns = theoretical_hourly_df.select_dtypes(include=['number']).columns.tolist()
        
        if 'dc_output' in numeric_columns:
            hourly_comparison['Theoretical DC Output (kW)'] = theoretical_hourly_df['dc_output']

        # Cap theoretical at the rolling max DC observed from the inverter.
        # During clipping hours both theoretical and actual hit the same ceiling,
        # so those hours contribute ~0 to the daily loss — no hours are discarded.
        # max_total_dc_power is in Watts (same unit as sensor.sg_total_dc_power).
        max_dc_kw = (max_total_dc_power / 1000) if max_total_dc_power > 0 else 0
        if max_dc_kw > 0 and 'Theoretical DC Output (kW)' in hourly_comparison.columns:
            hourly_comparison['Theoretical DC Capped (kW)'] = (
                hourly_comparison['Theoretical DC Output (kW)'].clip(upper=max_dc_kw)
            )
        else:
            hourly_comparison['Theoretical DC Capped (kW)'] = hourly_comparison.get(
                'Theoretical DC Output (kW)', pd.Series(dtype='float64')
            )

    if actual_hourly_df is not None and not actual_hourly_df.empty:
        for idx in hourly_comparison.index:
            if idx in actual_hourly_df.index:
                if 'Total DC Power (W)' in actual_hourly_df.columns:
                    hourly_comparison.loc[idx, 'Actual DC Power (kW)'] = actual_hourly_df.loc[idx, 'Total DC Power (W)'] / 1000
                
                if 'MPPT1 Voltage' in actual_hourly_df.columns: 
                    hourly_comparison.loc[idx, 'MPPT1 Voltage'] = actual_hourly_df.loc[idx, 'MPPT1 Voltage']
                if 'MPPT2 Voltage' in actual_hourly_df.columns:
                    hourly_comparison.loc[idx, 'MPPT2 Voltage'] = actual_hourly_df.loc[idx, 'MPPT2 Voltage']
                if 'MPPT1 Current' in actual_hourly_df.columns:
                    hourly_comparison.loc[idx, 'MPPT1 Current'] = actual_hourly_df.loc[idx, 'MPPT1 Current']
                if 'MPPT2 Current' in actual_hourly_df.columns:
                    hourly_comparison.loc[idx, 'MPPT2 Current'] = actual_hourly_df.loc[idx, 'MPPT2 Current']
    
    if hourly_comparison.empty:
        return hourly_comparison
    
    if 'Theoretical DC Output (kW)' in hourly_comparison.columns and 'Actual DC Power (kW)' in hourly_comparison.columns:
        hourly_comparison['MPPT Current Difference (%)'] = pd.Series(dtype='float64')

        mask_valid_mppt_c = (
            hourly_comparison['MPPT1 Current'].notna() & 
            hourly_comparison['MPPT2 Current'].notna() & 
            (hourly_comparison['MPPT1 Current'] > 0) & 
            (hourly_comparison['MPPT2 Current'] > 0)
        )
        
        hourly_comparison.loc[mask_valid_mppt_c, 'MPPT Current Difference (%)'] = (
            abs(hourly_comparison.loc[mask_valid_mppt_c, 'MPPT1 Current'] - 
                hourly_comparison.loc[mask_valid_mppt_c, 'MPPT2 Current']) / 
            ((hourly_comparison.loc[mask_valid_mppt_c, 'MPPT1 Current'] + 
              hourly_comparison.loc[mask_valid_mppt_c, 'MPPT2 Current']) / 2) * 100
        ).round(2)
        
        # Per-MPPT actual power from V×I sensors (more accurate than total/2)
        mask_mppt_power = (
            hourly_comparison['MPPT1 Voltage'].notna() &
            hourly_comparison['MPPT1 Current'].notna() &
            hourly_comparison['MPPT2 Voltage'].notna() &
            hourly_comparison['MPPT2 Current'].notna() &
            (hourly_comparison['MPPT1 Current'] > QUALITY_MIN_MPPT_CURRENT) &
            (hourly_comparison['MPPT2 Current'] > QUALITY_MIN_MPPT_CURRENT)
        )
        hourly_comparison['MPPT1 Power (kW)'] = pd.Series(dtype='float64')
        hourly_comparison['MPPT2 Power (kW)'] = pd.Series(dtype='float64')
        hourly_comparison.loc[mask_mppt_power, 'MPPT1 Power (kW)'] = (
            hourly_comparison.loc[mask_mppt_power, 'MPPT1 Voltage'] *
            hourly_comparison.loc[mask_mppt_power, 'MPPT1 Current'] / 1000
        ).round(3)
        hourly_comparison.loc[mask_mppt_power, 'MPPT2 Power (kW)'] = (
            hourly_comparison.loc[mask_mppt_power, 'MPPT2 Voltage'] *
            hourly_comparison.loc[mask_mppt_power, 'MPPT2 Current'] / 1000
        ).round(3)

        # Shoulder window: quality filter for shading (localized loss).
        # Soiling uses the full-day energy summation method and does not depend on this window.
        shoulder_lower = inverter_capacity_kw * SHOULDER_LOWER_RATIO
        shoulder_upper = inverter_capacity_kw * SHOULDER_UPPER_RATIO
        hourly_comparison['Shoulder Window'] = (
            (hourly_comparison['Theoretical DC Output (kW)'] >= shoulder_lower) &
            (hourly_comparison['Theoretical DC Output (kW)'] < shoulder_upper)
        )

        # Per-MPPT loss as % of per-MPPT theoretical (equal-string assumption)
        P_theo_each = hourly_comparison['Theoretical DC Output (kW)'] / 2

        mask_decomp = (
            mask_mppt_power &
            (P_theo_each > 0) &
            hourly_comparison['MPPT1 Power (kW)'].notna() &
            hourly_comparison['MPPT2 Power (kW)'].notna()
        )

        loss1_pct = pd.Series(np.nan, index=hourly_comparison.index)
        loss2_pct = pd.Series(np.nan, index=hourly_comparison.index)
        loss1_pct[mask_decomp] = (
            (P_theo_each[mask_decomp] - hourly_comparison.loc[mask_decomp, 'MPPT1 Power (kW)']) /
            P_theo_each[mask_decomp] * 100
        ).clip(lower=0)
        loss2_pct[mask_decomp] = (
            (P_theo_each[mask_decomp] - hourly_comparison.loc[mask_decomp, 'MPPT2 Power (kW)']) /
            P_theo_each[mask_decomp] * 100
        ).clip(lower=0)

        # Differential: extra loss on the worse MPPT → shading candidate
        differential_pct = pd.Series(np.nan, index=hourly_comparison.index)
        differential_pct[mask_decomp] = (loss1_pct[mask_decomp] - loss2_pct[mask_decomp]).abs()

        # Shading Loss: differential / 2 (one of two equal strings affected).
        # Restricted to shoulder-window hours to keep comparisons clean.
        hourly_comparison['Shading Loss (%)'] = pd.Series(dtype='float64')
        mask_valid_shading = mask_decomp & hourly_comparison['Shoulder Window']
        hourly_comparison.loc[mask_valid_shading, 'Shading Loss (%)'] = (
            differential_pct[mask_valid_shading] / 2
        ).clip(upper=50).round(2)
        
        hourly_comparison['date'] = hourly_comparison.index.date

        hourly_comparison['Daily Soiling Loss (%)'] = pd.Series(pd.NA, index=hourly_comparison.index)
        hourly_comparison['Daily Averaged Shading Loss (%)'] = pd.Series(pd.NA, index=hourly_comparison.index)

        for date, group in hourly_comparison.groupby('date'):
            try:
                last_ts = group.index.max()

                # Soiling: daily energy summation — capped theoretical vs actual.
                # Hours where MPPT current imbalance > 5% are shading-contaminated
                # and excluded from both sums. Hours with missing MPPT data are
                # included (cannot determine shading status).
                mppt_diff_ok = (
                    group['MPPT Current Difference (%)'].isna() |
                    (group['MPPT Current Difference (%)'] <= SHADING_EXCLUSION_MPPT_DIFF)
                )
                valid_hours = (
                    group['Actual DC Power (kW)'].notna() &
                    (group['Actual DC Power (kW)'] >= 0) &
                    mppt_diff_ok
                )
                if valid_hours.any():
                    theo_energy = group.loc[valid_hours, 'Theoretical DC Capped (kW)'].sum()
                    actual_energy = group.loc[valid_hours, 'Actual DC Power (kW)'].sum()
                    if theo_energy > 0:
                        soiling_loss = max(1.0, (theo_energy - actual_energy) / theo_energy * 100)
                        hourly_comparison.loc[last_ts, 'Daily Soiling Loss (%)'] = round(soiling_loss, 2)

                # Shading: daily average of per-hour differential losses with outlier removal
                valid_shading = group[group['Shading Loss (%)'].notna()]['Shading Loss (%)']
                if not valid_shading.empty and len(valid_shading) >= MIN_DAILY_DATA_POINTS:
                    median = valid_shading.median()
                    std = valid_shading.std()
                    if pd.notna(std) and std > 0:
                        filtered_shading = valid_shading[
                            (valid_shading >= median - 2 * std) &
                            (valid_shading <= median + 2 * std)
                        ]
                    else:
                        filtered_shading = valid_shading
                    if not filtered_shading.empty:
                        hourly_comparison.loc[last_ts, 'Daily Averaged Shading Loss (%)'] = round(filtered_shading.mean(), 2)

            except Exception as e:
                print(f"[WARNING] Error calculating daily losses for {date}: {e}")
    
    desired_order = [
        'Theoretical DC Output (kW)',
        'Theoretical DC Capped (kW)',
        'Actual DC Power (kW)',
        'MPPT1 Voltage',
        'MPPT2 Voltage',
        'MPPT1 Current',
        'MPPT2 Current',
        'MPPT Current Difference (%)',
        'Shading Loss (%)',
        'Daily Soiling Loss (%)',
        'Daily Averaged Shading Loss (%)',
        'date'
    ]

    existing_cols = [col for col in desired_order if col in hourly_comparison.columns]
    hourly_comparison = hourly_comparison[existing_cols]
    
    return hourly_comparison

def fetch_user_name_from_ha():
    """Fetch input_text.user_name from Ebsher to use as the Excel output folder."""
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    try:
        url = _api("/api/states/input_text.user_name")
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        name = resp.json().get("state", "").strip()
        if name and name not in ("unknown", "unavailable", ""):
            safe_name = re.sub(r'[^\w\-_. ]', '_', name).strip()
            print(f"[OKEY] Output folder set to: '{safe_name}'")
            return safe_name
    except Exception as e:
        print(f"[WARNING] Could not fetch user_name from HA: {e}")
    print("[INFO] Using default output folder: reports")
    return "reports"

def update_daily_soiling_loss_to_ha(loss_value, date_str):
    """Update Home Assistant input_text.daily_soiling_loss"""
    entity_id = "input_text.daily_soiling_loss"
    url = _api("/api/services/input_text/set_value")
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "entity_id": entity_id,
        "value": f"{loss_value:.2f}"
    }

    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=20)
        resp.raise_for_status()
        print(f"[OKEY] Updated {entity_id} in Home Assistant with value={loss_value:.2f} for date {date_str}")
    except Exception as e:
        print(f"[WARNING] Failed to update {entity_id} in Home Assistant: {e}")

def export_to_excel(
    all_results,
    hourly_averages,
    theoretical_detailed_df,
    theoretical_hourly_df,
    actual_minute_df=None,
    actual_hourly_df=None,
    hourly_comparison_df=None,
    filename="solar_analysis_combined.xlsx",
):
    """Export all relevant data to an Excel file"""
    try:
        import openpyxl

        df_detailed = pd.DataFrame(all_results or [])
        df_hourly = pd.DataFrame(hourly_averages or [])

        if actual_minute_df is not None and not actual_minute_df.empty:
            if isinstance(actual_minute_df.index, pd.DatetimeIndex) and actual_minute_df.index.tz is not None:
                actual_minute_df = actual_minute_df.copy()
                actual_minute_df.index = actual_minute_df.index.tz_localize(None)

        if actual_hourly_df is not None and not actual_hourly_df.empty:
            if isinstance(actual_hourly_df.index, pd.DatetimeIndex) and actual_hourly_df.index.tz is not None:
                actual_hourly_df = actual_hourly_df.copy()
                actual_hourly_df.index = actual_hourly_df.index.tz_localize(None)

        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            if not df_detailed.empty:
                df_detailed.to_excel(writer, sheet_name="Theoretical Detailed", index=False)
            if not df_hourly.empty:
                df_hourly.to_excel(writer, sheet_name="Theoretical Hourly", index=False)

            if actual_minute_df is not None and not actual_minute_df.empty:
                actual_minute_df.to_excel(writer, sheet_name="Actual Minute by Minute")
            if actual_hourly_df is not None and not actual_hourly_df.empty:
                actual_hourly_df.to_excel(writer, sheet_name="Actual Hourly")

            if hourly_comparison_df is not None and not hourly_comparison_df.empty:
                hourly_comparison_df.to_excel(writer, sheet_name="Hourly Comparison")

        return True

    except Exception as e:
        print(f"Error exporting to Excel: {e}")
        return None

def main():
    """Main function"""
    print(f"=== Starting Solar Analysis for {FORECAST_DATE} ===")
    
    # Step 1: Fetch configuration from Home Assistant
    fetch_configuration_from_ha()
    
    # Step 2: Get altitude
    altitude = get_altitude(LATITUDE, LONGITUDE)
    global ALTITUDE
    ALTITUDE = altitude
    
    # Step 3:  Fetch solar forecast data
    print("\n=== PHASE 1: FETCHING WEATHER DATA ===")
    solar_data = fetch_solar_forecast(FORECAST_DATE)
    
    # Step 4: Fetch weather data
    weather_data = fetch_weather_data(FORECAST_DATE)
    
    # Step 6: Merge data
    merged_weather_data = merge_data(weather_data, solar_data, altitude)
    
    # Step 7: Run theoretical calculations
    print("\n=== PHASE 3: RUNNING THEORETICAL CALCULATIONS ===")
    all_results, hourly_averages = run_theoretical_calculations(merged_weather_data)
    
    # Step 8: Get date range
    start_date, end_date = get_date_range_from_weather_data(merged_weather_data)
    
    if not start_date or not end_date: 
        print("Could not determine date range from weather data")
        return
    
    # Create output directory: reports/{client_name}/
    import os
    client_name = fetch_user_name_from_ha()
    report_dir = os.path.join("reports", client_name)
    os.makedirs(report_dir, exist_ok=True)

    output_filename = os.path.join(report_dir, f"solar_analysis_{start_date.date()}_to_{end_date.date()}.xlsx")
    
    # Step 9: Convert theoretical results to DataFrames
    theoretical_detailed_df, theoretical_hourly_df = convert_theoretical_to_dataframe(all_results, hourly_averages)
    
    # Step 10: Fetch actual data from Home Assistant
    print("\n=== PHASE 4: FETCHING ACTUAL DATA FROM HOME ASSISTANT ===")
    actual_minute_df, actual_hourly_df = fetch_home_assistant_data_range(start_date, end_date)
    
    # Step 11: Create comparison
    print("\n=== PHASE 5: CREATING COMPARISON ANALYSIS ===")
    hourly_comparison_df = None
    try:
        hourly_comparison_df = create_complete_comparison(
            theoretical_detailed_df,
            theoretical_hourly_df,
            actual_minute_df,
            actual_hourly_df,
            inverter_capacity_kw=INVERTER_CAPACITY_KW,
            max_total_dc_power=MAX_TOTAL_DC_POWER,
        )

        if hourly_comparison_df is None:
            print("[INFO] No comparison DataFrame returned.")

        elif hourly_comparison_df.empty:
            print("[INFO] Comparison DataFrame is empty.")

        else:
            print(f"[INFO] Comparison DataFrame has {len(hourly_comparison_df)} rows.")

            # Display Daily Averaged Losses
            print("\n=== DAILY AVERAGED LOSSES ===")

            # Print Shading Losses
            shading_col = "Daily Averaged Shading Loss (%)"
            if shading_col in hourly_comparison_df.columns:
                daily_shading = hourly_comparison_df[hourly_comparison_df[shading_col].notna()]
                if not daily_shading.empty:
                    print(f"\n{'Date':<12} {shading_col}")
                    print("=============================================")
                    for idx, row in daily_shading.iterrows():
                        date_str = idx.strftime("%Y-%m-%d")
                        loss_value = row[shading_col]
                        print(f"{date_str:<12} {loss_value}")

            # Print Soiling Losses
            soiling_col = "Daily Soiling Loss (%)"
            if soiling_col in hourly_comparison_df.columns:
                daily_soiling = hourly_comparison_df[hourly_comparison_df[soiling_col].notna()]
                if not daily_soiling.empty:
                    print(f"\n{'Date':<12} {soiling_col}")
                    print("=============================================")
                    for idx, row in daily_soiling.iterrows():
                        date_str = idx.strftime("%Y-%m-%d")
                        loss_value = row[soiling_col]
                        print(f"{date_str:<12} {loss_value}")
                    
                    # Update HA with soiling loss (not shading)
                    last_idx, last_row = list(daily_soiling.tail(1).iterrows())[0]
                    last_date_str = last_idx.strftime("%Y-%m-%d")
                    last_loss_value = last_row[soiling_col]
                    update_daily_soiling_loss_to_ha(last_loss_value, last_date_str)

                else:
                    print("[INFO] No daily averaged soiling losses found.")

            else:
                print(f"[INFO] Column '{soiling_col}' not found in hourly_comparison_df.")

    except Exception as e:
        print(f"Error creating comparison: {e}")
        hourly_comparison_df = pd.DataFrame()
    
    # Step 12: Export all data to Excel
    print("\n=== PHASE 6: EXPORTING RESULTS TO EXCEL ===")
    result = export_to_excel(
        all_results,
        hourly_averages,
        theoretical_detailed_df,
        theoretical_hourly_df,
        actual_minute_df,
        actual_hourly_df,
        hourly_comparison_df,
        output_filename
    )
    
    if result: 
        print(f"\n[OKEY] Analysis complete! Results saved to {output_filename}")
    else:
        print("\n[INFO] Error exporting results to Excel")

if __name__ == "__main__": 
    main()
