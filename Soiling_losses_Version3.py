import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
import pvlib  # Added missing import
from pvlib.solarposition import get_solarposition
from scipy.interpolate import PchipInterpolator
import pytz
from collections import defaultdict
import re  # For string manipulation
from zoneinfo import ZoneInfo
from astral.location import LocationInfo
from astral.sun import sun
from urllib.parse import quote
import sys


# === CONFIGURATION ===
# Default values (will be updated from Home Assistant)
LATITUDE = 0
LONGITUDE = 0
TIMEZONE = 'Asia/Amman'
API_KEY = "4d9f599d094608656284561fba4a79f7"  # OpenWeather API key

# System parameters (will be updated from Home Assistant)
TILT = 0
AZIMUTH = 0
ALBEDO = 0.2
PANEL_PEAK_POWER = 0
NUMBER_OF_PANELS = 0
TEMP_COEFFICIENT = -0.0035
IAM_ANGLES = [0.0, 25.0, 45.0, 60.0, 65.0, 70.0, 75.0, 80.0, 90.0]
IAM_VALUES = [1.000, 1.000, 0.995, 0.962, 0.936, 0.903, 0.851, 0.754, 0.000]
INVERTER_CAPACITY_KW = 0

# Date for forecast

ALTITUDE = 770  # Default altitude, will be updated by get_altitude()

# Home Assistant configuration
HOME_ASSISTANT_URL = "http://default-ha-url"
ACCESS_TOKEN = "default-token"
FORECAST_DATE = "2025-07-16"

if len(sys.argv) >= 4:
    HOME_ASSISTANT_URL = sys.argv[1]
    ACCESS_TOKEN = sys.argv[2]
    FORECAST_DATE = sys.argv[3]
    print(f"[INFO] Using HA URL: {HOME_ASSISTANT_URL}")
    print(f"[INFO] Using forecast date: {FORECAST_DATE}")
else:
    print("[WARNING] Using default hardcoded values.")

SENSOR_DEFINITIONS = {
    'MPPT1 Voltage': "sensor.mppt1_voltage",
    'MPPT2 Voltage': "sensor.mppt2_voltage",
    'MPPT1 Current': "sensor.mppt1_current",
    'MPPT2 Current': "sensor.mppt2_current",
    'AC Output': "sensor.total_ac_power",
    'DC Output': "sensor.total_dc_power"
}

# Home Assistant sensor mapping for configuration variables
HA_CONFIG_SENSORS = {
    'LATITUDE': "input_text.solar_system_latitude",
    'LONGITUDE': "input_text.solar_system_longitude",
    'TILT': "input_text.solar_panel_tilt",
    'AZIMUTH': "input_text.solar_panel_azimuth",
    'PANEL_PEAK_POWER': "input_text.solar_panel_info",
    'NUMBER_OF_PANELS': "input_text.solar_panel_count",
    'INVERTER_CAPACITY_KW': "input_text.solar_inverter_info"
}

# Function to fetch configuration values from Home Assistant
def fetch_configuration_from_ha():
    """Fetch configuration values from Home Assistant sensors"""
    global LATITUDE, LONGITUDE, TILT, AZIMUTH, PANEL_PEAK_POWER, NUMBER_OF_PANELS, INVERTER_CAPACITY_KW
    
    print("\n=== Fetching Configuration from Home Assistant ===")
    
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Fetch each sensor value
    for config_name, sensor_entity_id in HA_CONFIG_SENSORS.items():
        try:
            url = f"{HOME_ASSISTANT_URL}/api/states/{sensor_entity_id}"
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                sensor_data = response.json()
                value = sensor_data.get('state')
                
                # Convert to appropriate type
                if value not in ('unknown', 'unavailable'):
                    if config_name in ('LATITUDE', 'LONGITUDE', 'TILT', 'AZIMUTH', 'INVERTER_CAPACITY_KW'):
                        value = float(value)
                    elif config_name in ('PANEL_PEAK_POWER', 'NUMBER_OF_PANELS'):
                        value = int(value)
                    
                    # Update the global variable
                    globals()[config_name] = value
                    print(f"[OKEY] {config_name} updated to: {value}")
                else:
                    print(f"[WARRNING] {config_name} sensor returned: {value}, using default value: {globals()[config_name]}")
            else:
                print(f"[WARRNING] Failed to fetch {config_name} from sensor {sensor_entity_id}: {response.status_code}")
                print(f"   Using default value: {globals()[config_name]}")
                
        except Exception as e:
            print(f"[INFO] Error fetching {config_name}: {e}")
            print(f"   Using default value: {globals()[config_name]}")
    
    # Print summary of all configuration values
    print("\n=== Configuration Values Summary ===")
    print(f"LATITUDE: {LATITUDE}")
    print(f"LONGITUDE: {LONGITUDE}")
    print(f"TILT: {TILT}")
    print(f"AZIMUTH: {AZIMUTH}")
    print(f"PANEL_PEAK_POWER: {PANEL_PEAK_POWER}")
    print(f"NUMBER_OF_PANELS: {NUMBER_OF_PANELS}")
    print(f"INVERTER_CAPACITY_KW: {INVERTER_CAPACITY_KW}")
    print(f"TIMEZONE: {TIMEZONE}")
    print(f"ALBEDO: {ALBEDO}")
    print(f"TEMP_COEFFICIENT: {TEMP_COEFFICIENT}")
    print("=======================================\n")

# ===== WEATHER DATA FETCHING FUNCTIONS =====
def get_altitude(lat, lon):
    """Fetch altitude (elevation) for given coordinates in meters"""
    print(f"[INFO] Fetching altitude data for coordinates ({lat}, {lon})...")
    
    try:
        # Using Open-Topo-Data API which is free and doesn't require authentication
        url = f"https://api.opentopodata.org/v1/aster30m?locations={lat},{lon}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("results") and len(data["results"]) > 0:
                elevation = data["results"][0]["elevation"]
                print(f"[OKEY] Altitude retrieved: {elevation} meters")
                return elevation
        
        # Fallback if the API request fails
        print("[WARRNING] Could not retrieve altitude from API, using default value")
        return 770  # Default altitude
    
    except Exception as e:
        print(f"[WARRNING] Error retrieving altitude: {e}")
        print("Using default altitude value")
        return 770  # Default altitude

def fetch_solar_forecast(date):
    """Fetch solar irradiance forecast data from OpenWeatherMap API"""
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
        print(f"[INFO] Solar API request failed: {solar_response.status_code}")
        print(solar_response.text)
        raise Exception("Failed to fetch solar forecast data")

def fetch_weather_data(date):
    """Fetch historical weather data from OpenWeatherMap API for each hour of the day"""
    print(f"[INFO] Fetching Weather Data for {date}...")
    
    # Prepare date and hourly timestamps
    date_obj = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    timestamps = [
        int((date_obj + timedelta(hours=h)).timestamp())
        for h in range(24)
    ]
    
    # Data collector
    weather_data = []
    
    # Fetch and format data for each hour
    for ts in timestamps:
        utc_dt = datetime.fromtimestamp(ts, tz=timezone.utc)  # Updated to avoid deprecation warning
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
                
                # Extract relevant fields
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
            print(f"Failed for {timestamp_str}: {response.status_code} - {response.text}")
    
    print(f"[OKEY] Successfully retrieved weather data for {len(weather_data)} hours.")
    return weather_data

def merge_data(weather_data, solar_forecast, altitude):
    """Merge weather and solar forecast data, adding solar position information"""
    print("[INFO] Merging data and calculating solar position...")
    
    # Build forecast map (Shift timestamps back by 3 hours)
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
    
    # Merge data
    merged_output = []
    
    for entry in weather_data:
        timestamp_utc = pd.Timestamp(entry["timestamp_utc"]).tz_convert("UTC")
        local_time = timestamp_utc.tz_convert(TIMEZONE)
        
        # "time": Start of the hour
        time_str = local_time.strftime("%H:%M:%S")
        
        # "actual_time": Mid of the hour
        actual_time = local_time.replace(minute=30, second=0, microsecond=0)
        actual_time_str = actual_time.strftime("%H:%M:%S")
        
        # Solar zenith angle
        solpos = get_solarposition(
            time=actual_time,
            latitude=LATITUDE,
            longitude=LONGITUDE,
            altitude=altitude
        )
        zenith = round(solpos["zenith"].iloc[0], 2)
        
        # Irradiance from forecast
        forecast = forecast_map.get(timestamp_utc, {"ghi": 0, "dni": 0, "dhi": 0})
        
        merged_output.append({
            "date": entry["date"],
            "time": time_str,
            "timestamp_utc": entry["timestamp_utc"],
            "ghi": forecast["ghi"],
            "dni": forecast["dni"],
            "dhi": forecast["dhi"],
            "wind_speed": entry["wind_speed"],
            "ambient_temp": entry["ambient_temp"],
            "zenith_angle": zenith,
            "actual_time": actual_time_str
        })
    
    return merged_output

# ===== SOILING ANALYSIS FUNCTIONS =====
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

    isotropic = perez['isotropic']
    circumsolar = perez['circumsolar']
    horizon = perez['horizon']

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

def get_date_range_from_weather_data(weather_data):
    dates = [entry['date'] for entry in weather_data]
    unique_dates = sorted(set(dates))
    
    if not unique_dates:
        return None, None
    
    # Convert to datetime objects
    start_date = datetime.strptime(unique_dates[0], "%Y-%m-%d")
    end_date = datetime.strptime(unique_dates[-1], "%Y-%m-%d")
    
    return start_date, end_date

def calculate_hourly_averages(all_results):
    # Dictionary to store data grouped by date and hour
    hourly_data = defaultdict(lambda: defaultdict(list))
    
    # Group data by date and hour
    for result in all_results:
        date = result['date']
        hour = int(result['actual_time'].split(':')[0])
        
        # Group by date and hour
        key = f"{date}_{hour:02d}"
        
        hourly_data[key]['ghi'].append(result['ghi'])
        hourly_data[key]['dni'].append(result['dni'])
        hourly_data[key]['poa'].append(result['poa'])
        hourly_data[key]['tcell'].append(result['tcell'])
        hourly_data[key]['dc_output'].append(result['dc_output'])
        hourly_data[key]['date'] = date
        hourly_data[key]['hour'] = hour
        
    # Calculate averages for each hour
    hourly_averages = []
    
    for key, data in hourly_data.items():
        averages = {
            'date': data['date'],
            'hour': data['hour'],
            'ghi': sum(data['ghi']) / len(data['ghi']),
            'dni': sum(data['dni']) / len(data['dni']),
            'poa': sum(data['poa']) / len(data['poa']),
            'tcell': sum(data['tcell']) / len(data['tcell']),
            'dc_output': sum(data['dc_output']) / len(data['dc_output'])
        }
        hourly_averages.append(averages)
    
    # Sort by date and hour
    hourly_averages.sort(key=lambda x: (x['date'], x['hour']))
    
    return hourly_averages

def format_datetime_windows_compatible(dt):
    # First format with standard format codes
    formatted = dt.strftime("%m/%d/%Y %I:%M:%S %p")
    
    # Remove leading zeros from month and day
    formatted = re.sub(r'0(\d)/', r'\1/', formatted)
    
    # Remove leading zero from hour (but not from minutes/seconds)
    formatted = re.sub(r' 0(\d):', r' \1:', formatted)
    
    return formatted

def create_datetime_column(data_list):
    for entry in data_list:
        # Parse the date (YYYY-MM-DD) and time
        year, month, day = map(int, entry['date'].split('-'))
        
        if 'actual_time' in entry:
            # For detailed results
            hour, minute, second = map(int, entry['actual_time'].split(':'))
        else:
            # For hourly averages
            hour = entry['hour']
            minute = 0
            second = 0
            
        # Create datetime object
        dt = datetime(year, month, day, hour, minute, second)
        
        # Format using our Windows-compatible function
        entry['formatted_datetime'] = format_datetime_windows_compatible(dt)
        
        # Also add a datetime object for later comparison
        entry['datetime_obj'] = dt
        
    return data_list

# === HOME ASSISTANT DATA FUNCTIONS ===
def get_ha_config():
    api_url = f"{HOME_ASSISTANT_URL}/api/config"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}", "content-type": "application/json"}
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return None

def get_sunrise_sunset(latitude, longitude, timezone_str, target_date):
    location = LocationInfo("My Home", "My Region", timezone_str, latitude, longitude)
    s = sun(location.observer, date=target_date, tzinfo=location.timezone)
    return s["sunrise"], s["sunset"]

def get_sensors_data(entity_ids, start_time, end_time):
    start_iso = start_time.isoformat()
    end_iso = end_time.isoformat()
    encoded_start_iso = quote(start_iso)
    api_url = f"{HOME_ASSISTANT_URL}/api/history/period/{encoded_start_iso}"
    params = {"filter_entity_id": ",".join(entity_ids), "end_time": end_iso}
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}", "content-type": "application/json"}
    try:
        response = requests.get(api_url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return None

def fetch_home_assistant_data_range(start_date, end_date):
    ha_config = get_ha_config()
    if not ha_config:
        return None, None

    ha_timezone_str = ha_config.get("time_zone", "UTC")
    local_timezone = ZoneInfo(ha_timezone_str)
    
    # Adjust end date to include the full day
    end_date = end_date + timedelta(days=1) - timedelta(seconds=1)
    
    entity_ids_to_fetch = list(SENSOR_DEFINITIONS.values())
    all_sensor_data = get_sensors_data(entity_ids_to_fetch, start_date, end_date)

    if not all_sensor_data:
        return None, None

    entity_id_to_friendly_name = {v: k for k, v in SENSOR_DEFINITIONS.items()}
    all_series = {}
    for entity_history in all_sensor_data:
        if not entity_history: continue
        entity_id = entity_history[0]['entity_id']
        friendly_name = entity_id_to_friendly_name.get(entity_id, entity_id)
        timestamps = [datetime.fromisoformat(state['last_changed']) for state in entity_history]
        values = [state['state'] for state in entity_history]
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

    # Format the dataframes for return
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
    
    # Store timezone info before converting to timezone-naive
    timezone_info = local_timezone if df_minute_for_excel.index.tzinfo else None
    
    # Convert to timezone-naive for Excel compatibility (but preserve original times)
    if timezone_info:
        df_minute_for_excel.index = df_minute_for_excel.index.tz_localize(None)
        df_hourly_for_excel.index = df_hourly_for_excel.index.tz_localize(None)
    
    # Return both dataframes
    return df_minute_for_excel, df_hourly_for_excel

def convert_theoretical_to_dataframe(all_results, hourly_averages):
    # Convert detailed results to DataFrame
    detailed_df = pd.DataFrame(all_results)
    
    # Create datetime index for detailed results
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
            # Use current datetime as fallback (will be filtered out later)
            datetime_indices.append(datetime.now())
    
    # Make sure we only keep numeric columns when we set the index
    numeric_cols = ['ghi', 'dni', 'poa', 'tcell', 'dc_output']
    for col in numeric_cols:
        if col in detailed_df.columns:
            detailed_df[col] = pd.to_numeric(detailed_df[col], errors='coerce')
    
    detailed_df['datetime'] = datetime_indices
    detailed_df.set_index('datetime', inplace=True)
    
    # Convert hourly averages to DataFrame
    hourly_df = pd.DataFrame(hourly_averages)
    
    # Create datetime index for hourly results
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
    
    # Convert numeric columns
    for col in numeric_cols:
        if col in hourly_df.columns:
            hourly_df[col] = pd.to_numeric(hourly_df[col], errors='coerce')
    
    hourly_df['datetime'] = hourly_indices
    hourly_df.set_index('datetime', inplace=True)
    
    return detailed_df, hourly_df

def create_complete_comparison(theoretical_detailed_df, theoretical_hourly_df, 
                              actual_minute_df, actual_hourly_df, inverter_capacity_kw=INVERTER_CAPACITY_KW):
    # Create hourly comparison first (easier to align)
    hourly_comparison = pd.DataFrame()
    
    if theoretical_hourly_df is not None and not theoretical_hourly_df.empty:
        # Filter to only include numeric columns for calculations
        numeric_columns = theoretical_hourly_df.select_dtypes(include=['number']).columns.tolist()
        
        # Add theoretical data (only numeric columns)
        if 'dc_output' in numeric_columns:
            hourly_comparison['Theoretical DC Output (kW)'] = theoretical_hourly_df['dc_output']
    
    if actual_hourly_df is not None and not actual_hourly_df.empty:
        # Add actual data - first reindex to match hourly comparison
        for idx in hourly_comparison.index:
            if idx in actual_hourly_df.index:
                if 'Total DC Power (W)' in actual_hourly_df.columns:
                    hourly_comparison.loc[idx, 'Actual DC Power (kW)'] = actual_hourly_df.loc[idx, 'Total DC Power (W)'] / 1000
                
                # Add MPPT values
                if 'MPPT1 Voltage' in actual_hourly_df.columns:
                    hourly_comparison.loc[idx, 'MPPT1 Voltage'] = actual_hourly_df.loc[idx, 'MPPT1 Voltage']
                if 'MPPT2 Voltage' in actual_hourly_df.columns:
                    hourly_comparison.loc[idx, 'MPPT2 Voltage'] = actual_hourly_df.loc[idx, 'MPPT2 Voltage']
                if 'MPPT1 Current' in actual_hourly_df.columns:
                    hourly_comparison.loc[idx, 'MPPT1 Current'] = actual_hourly_df.loc[idx, 'MPPT1 Current']
                if 'MPPT2 Current' in actual_hourly_df.columns:
                    hourly_comparison.loc[idx, 'MPPT2 Current'] = actual_hourly_df.loc[idx, 'MPPT2 Current']
    
    # Check if we have data to work with
    if hourly_comparison.empty:
        return hourly_comparison  # Return empty DataFrame
    
    # Calculate percentage difference
    if 'Theoretical DC Output (kW)' in hourly_comparison.columns and 'Actual DC Power (kW)' in hourly_comparison.columns:
        # Calculate absolute difference
        hourly_comparison['DC Output Difference (kW)'] = hourly_comparison['Theoretical DC Output (kW)'] - hourly_comparison['Actual DC Power (kW)']
        
        # Calculate percentage difference - avoid division by zero
        mask = hourly_comparison['Theoretical DC Output (kW)'] > 0
        hourly_comparison['DC Output Difference (%)'] = pd.Series(dtype='float64')  # Initialize empty series
        hourly_comparison.loc[mask, 'DC Output Difference (%)'] = (
            (hourly_comparison.loc[mask, 'Theoretical DC Output (kW)'] - hourly_comparison.loc[mask, 'Actual DC Power (kW)']) / 
            hourly_comparison.loc[mask, 'Theoretical DC Output (kW)'] * 100
        ).round(2)
        
        # Convert columns to object type before adding string values
        hourly_comparison['Theoretical DC Output Limited (kW)'] = hourly_comparison['Theoretical DC Output (kW)'].astype('object')
        hourly_comparison['Actual DC Power Limited (kW)'] = hourly_comparison['Actual DC Power (kW)'].astype('object')
        
        # Identify rows where values exceed inverter capacity
        mask_theo_over = hourly_comparison['Theoretical DC Output (kW)'] > inverter_capacity_kw
        mask_actual_over = hourly_comparison['Actual DC Power (kW)'] > inverter_capacity_kw
        
        # Mark as N/A
        hourly_comparison.loc[mask_theo_over, 'Theoretical DC Output Limited (kW)'] = 'N/A'
        hourly_comparison.loc[mask_actual_over, 'Actual DC Power Limited (kW)'] = 'N/A'
        
        # If either is N/A, make both N/A
        mask_any_na = (hourly_comparison['Theoretical DC Output Limited (kW)'] == 'N/A') | (hourly_comparison['Actual DC Power Limited (kW)'] == 'N/A')
        hourly_comparison.loc[mask_any_na, 'Theoretical DC Output Limited (kW)'] = 'N/A'
        hourly_comparison.loc[mask_any_na, 'Actual DC Power Limited (kW)'] = 'N/A'
        
        # Calculate MPPT differences (where we have valid data)
        hourly_comparison['MPPT Voltage Difference (%)'] = pd.Series(dtype='float64')
        hourly_comparison['MPPT Current Difference (%)'] = pd.Series(dtype='float64')
        
        # Calculate MPPT voltage and current percentage differences
        mask_valid_mppt_v = (
            hourly_comparison['MPPT1 Voltage'].notna() & 
            hourly_comparison['MPPT2 Voltage'].notna() & 
            (hourly_comparison['MPPT1 Voltage'] > 0) & 
            (hourly_comparison['MPPT2 Voltage'] > 0)
        )
        
        hourly_comparison.loc[mask_valid_mppt_v, 'MPPT Voltage Difference (%)'] = (
            abs(hourly_comparison.loc[mask_valid_mppt_v, 'MPPT1 Voltage'] - 
                hourly_comparison.loc[mask_valid_mppt_v, 'MPPT2 Voltage']) / 
            ((hourly_comparison.loc[mask_valid_mppt_v, 'MPPT1 Voltage'] + 
              hourly_comparison.loc[mask_valid_mppt_v, 'MPPT2 Voltage']) / 2) * 100
        ).round(2)
        
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
        
        # Initialize balanced MPPT columns with 'N/A' as default
        hourly_comparison['Theoretical DC (Balanced MPPTs)'] = pd.Series('N/A', index=hourly_comparison.index, dtype='object')
        hourly_comparison['Actual DC (Balanced MPPTs)'] = pd.Series('N/A', index=hourly_comparison.index, dtype='object')
        hourly_comparison['DC Difference (Balanced MPPTs) (%)'] = pd.Series('N/A', index=hourly_comparison.index, dtype='object')
        
        # Calculate 20% threshold of inverter capacity
        capacity_threshold = inverter_capacity_kw * 0.2
        
        # Only populate balanced MPPT columns when differences are less than 5%
        # AND both values are higher than 20% of inverter capacity
        mask_balanced = (
            (hourly_comparison['Theoretical DC Output Limited (kW)'] != 'N/A') & 
            (hourly_comparison['Actual DC Power Limited (kW)'] != 'N/A') &
            hourly_comparison['MPPT Voltage Difference (%)'].notna() &
            hourly_comparison['MPPT Current Difference (%)'].notna() &
            (hourly_comparison['MPPT Voltage Difference (%)'] < 5) &
            (hourly_comparison['MPPT Current Difference (%)'] < 5) &
            (pd.to_numeric(hourly_comparison['Theoretical DC Output Limited (kW)'], errors='coerce') > capacity_threshold) &
            (pd.to_numeric(hourly_comparison['Actual DC Power Limited (kW)'], errors='coerce') > capacity_threshold)
        )
        
        # Only for balanced MPPTs with difference < 5% and above threshold, add values and calculate difference
        hourly_comparison.loc[mask_balanced, 'Theoretical DC (Balanced MPPTs)'] = hourly_comparison.loc[mask_balanced, 'Theoretical DC Output Limited (kW)']
        hourly_comparison.loc[mask_balanced, 'Actual DC (Balanced MPPTs)'] = hourly_comparison.loc[mask_balanced, 'Actual DC Power Limited (kW)']
        
        # Convert values to numeric for calculation, handling errors safely
        theo_balanced_numeric = pd.to_numeric(hourly_comparison.loc[mask_balanced, 'Theoretical DC (Balanced MPPTs)'], errors='coerce')
        actual_balanced_numeric = pd.to_numeric(hourly_comparison.loc[mask_balanced, 'Actual DC (Balanced MPPTs)'], errors='coerce')
        
        # Calculate difference only where both values are valid numbers
        valid_rows = theo_balanced_numeric.notna() & (theo_balanced_numeric > 0) & actual_balanced_numeric.notna()
        if not valid_rows.empty:
            diff_pct = ((theo_balanced_numeric[valid_rows] - actual_balanced_numeric[valid_rows]) / 
                        theo_balanced_numeric[valid_rows] * 100).round(2)
            
            # Convert back to string series with index for assignment
            diff_series = pd.Series('N/A', index=hourly_comparison.index, dtype='object')
            diff_series.loc[valid_rows.index[valid_rows]] = diff_pct.loc[valid_rows.index[valid_rows]].astype(str)
            hourly_comparison.loc[mask_balanced, 'DC Difference (Balanced MPPTs) (%)'] = diff_series.loc[mask_balanced]
        
        # Add daily averaged soiling losses - calculate for each day, show at last timestamp
        # First extract date from datetime index and create column
        hourly_comparison['date'] = hourly_comparison.index.date
        
        # Initialize column for daily averaged soiling losses
        hourly_comparison['Daily Averaged Soiling Losses (%)'] = pd.Series(pd.NA, index=hourly_comparison.index)
        
        # For each date, find valid DC Difference values and calculate average
        for date, group in hourly_comparison.groupby('date'):
            try:
                # Get only the values where DC Difference for Balanced MPPTs is not 'N/A'
                valid_diffs = group[group['DC Difference (Balanced MPPTs) (%)'] != 'N/A']['DC Difference (Balanced MPPTs) (%)']
                
                # Convert to numeric, handling errors
                numeric_diffs = pd.to_numeric(valid_diffs, errors='coerce')
                
                if not numeric_diffs.empty and not numeric_diffs.isna().all():
                    avg_diff = numeric_diffs.mean()
                    
                    # Find the last timestamp for this date
                    last_ts = group.index.max()
                    
                    # Set the average at this timestamp
                    hourly_comparison.loc[last_ts, 'Daily Averaged Soiling Losses (%)'] = round(avg_diff, 2)
            except Exception:
                pass
    
    # Return the hourly comparison dataframe
    return hourly_comparison

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
    """
    Export all relevant data to an Excel file.
    This version is defensive: it only reorders columns if
    'formatted_datetime' actually exists.
    """
    try:
        import openpyxl  # ensure engine is available

        # --- Build DataFrames from theoretical lists ---
        if all_results is not None:
            all_results = create_datetime_column(all_results)
        if hourly_averages is not None:
            hourly_averages = create_datetime_column(hourly_averages)

        df_detailed = pd.DataFrame(all_results or [])
        df_hourly = pd.DataFrame(hourly_averages or [])

        # --- Reorder columns only if 'formatted_datetime' exists ---
        if "formatted_datetime" in df_detailed.columns:
            detailed_cols = ["formatted_datetime"] + [
                col
                for col in df_detailed.columns
                if col not in ("formatted_datetime", "datetime_obj")
            ]
            df_detailed = df_detailed[detailed_cols]

        if "formatted_datetime" in df_hourly.columns:
            hourly_cols = ["formatted_datetime"] + [
                col
                for col in df_hourly.columns
                if col not in ("formatted_datetime", "datetime_obj")
            ]
            df_hourly = df_hourly[hourly_cols]

        # --- Strip timezone from actual data indexes (Excel doesn't like tz-aware) ---
        if actual_minute_df is not None and not actual_minute_df.empty:
            if isinstance(actual_minute_df.index, pd.DatetimeIndex) and actual_minute_df.index.tz is not None:
                actual_minute_df = actual_minute_df.copy()
                actual_minute_df.index = actual_minute_df.index.tz_localize(None)

        if actual_hourly_df is not None and not actual_hourly_df.empty:
            if isinstance(actual_hourly_df.index, pd.DatetimeIndex) and actual_hourly_df.index.tz is not None:
                actual_hourly_df = actual_hourly_df.copy()
                actual_hourly_df.index = actual_hourly_df.index.tz_localize(None)

        # --- Write everything to Excel ---
        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            # Theoretical
            if not df_detailed.empty:
                df_detailed.to_excel(writer, sheet_name="Theoretical Detailed", index=False)
            if not df_hourly.empty:
                df_hourly.to_excel(writer, sheet_name="Theoretical Hourly", index=False)

            # Actual data
            if actual_minute_df is not None and not actual_minute_df.empty:
                actual_minute_df.to_excel(writer, sheet_name="Actual Minute by Minute")
            if actual_hourly_df is not None and not actual_hourly_df.empty:
                actual_hourly_df.to_excel(writer, sheet_name="Actual Hourly")

            # Comparison
            if hourly_comparison_df is not None and not hourly_comparison_df.empty:
                # Simple: just dump as-is. If you want custom ordering later, we can add it back safely.
                hourly_comparison_df.to_excel(writer, sheet_name="Hourly Comparison")

        return True

    except Exception as e:
        print(f"Error exporting to Excel: {e}")
        return None


def run_theoretical_calculations(weather_data_list):
    """Run theoretical calculations based on weather data"""
    # Store all results for later hourly averaging
    all_results = []
    
    # Process each time point in the JSON file
    for entry in weather_data_list:
        date = entry['date']
        actual_time = entry['actual_time']
        
        # Extract weather parameters
        ghi = entry['ghi']
        dni = entry['dni']
        dhi = entry['dhi']
        ambient_temp = entry['ambient_temp']
        wind_speed = entry['wind_speed']
        
        # Parse actual_time and create timestamp
        hour, minute, second = map(int, actual_time.split(':'))
        year, month, day = map(int, date.split('-'))
        
        # Create timestamp with actual time
        timestamp = pytz.timezone(TIMEZONE).localize(datetime(year, month, day, hour, minute, second))
        
        # Skip entries with zero or very low DNI (usually means it's night)
        if dni < 10:
            continue
        
        # Calculate irradiance and output
        results, weather_df = calculate_poa_irradiance_detailed(
            LATITUDE, LONGITUDE, TILT, AZIMUTH, timestamp,
            dni=dni, ghi=ghi, dhi=dhi, albedo=ALBEDO
        )
        
        tcell = calculate_tcell_faiman(ghi, ambient_temp, wind_speed)
        system_results = calculate_system_output_dc(results['poa_global_after_iam'], PANEL_PEAK_POWER, NUMBER_OF_PANELS, tcell, TEMP_COEFFICIENT)
        
        # Store results for hourly averaging
        all_results.append({
            'date': date,
            'actual_time': actual_time,
            'ghi': ghi,
            'dni': dni,
            'poa': results['poa_global_after_iam'],
            'tcell': tcell,
            'dc_output': system_results['e_array_actual']
        })
    
    # Calculate hourly averages
    hourly_averages = calculate_hourly_averages(all_results)
    
    return all_results, hourly_averages
    
def update_daily_soiling_loss_to_ha(loss_value, date_str):
    """
    Update Home Assistant input_text.daily_soiling_loss
    with the daily averaged soiling loss value.
    """
    entity_id = "input_text.daily_soiling_loss"

    url = f"{HOME_ASSISTANT_URL}/api/services/input_text/set_value"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "entity_id": entity_id,
        # HA expects a string value for input_text
        "value": f"{loss_value:.2f}"
    }

    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=20)
        resp.raise_for_status()
        print(
            f"[OKEY] Updated {entity_id} in Home Assistant with "
            f"value={loss_value:.2f} for date {date_str}"
        )
    except Exception as e:
        print(f"[WARNING] Failed to update {entity_id} in Home Assistant: {e}")

def main():
    """Main function to orchestrate the entire workflow"""
    print(f"=== Starting Combined Weather and Soiling Analysis for {FORECAST_DATE} ===")
    
    # New: First fetch configuration from Home Assistant
    fetch_configuration_from_ha()
    
    # Step 1: Get altitude for the location
    altitude = get_altitude(LATITUDE, LONGITUDE)
    global ALTITUDE
    ALTITUDE = altitude
    
    # Step 2: Fetch solar forecast data
    print("\n=== PHASE 1: FETCHING WEATHER AND SOLAR FORECAST DATA ===")
    solar_data = fetch_solar_forecast(FORECAST_DATE)
    
    # Step 3: Fetch weather data
    weather_data = fetch_weather_data(FORECAST_DATE)
    
    # Step 4: Merge data and add solar position
    merged_weather_data = merge_data(weather_data, solar_data, altitude)
    
    # Step 5: Run theoretical calculations based on the merged data
    print("\n=== PHASE 2: RUNNING THEORETICAL CALCULATIONS ===")
    all_results, hourly_averages = run_theoretical_calculations(merged_weather_data)
    
    # Step 6: Get date range from weather data
    start_date, end_date = get_date_range_from_weather_data(merged_weather_data)
    
    if not start_date or not end_date:
        print("Could not determine date range from weather data")
        return
    
    # Define output filename based on date range
    output_filename = f"solar_analysis_{start_date.date()}_to_{end_date.date()}.xlsx"
    
    # Step 7: Convert theoretical results to DataFrames with datetime indices
    theoretical_detailed_df, theoretical_hourly_df = convert_theoretical_to_dataframe(all_results, hourly_averages)
    
    # Step 8: Fetch actual data from Home Assistant for the same date range
    print("\n=== PHASE 3: FETCHING ACTUAL DATA FROM HOME ASSISTANT ===")
    actual_minute_df, actual_hourly_df = fetch_home_assistant_data_range(start_date, end_date)
    
    # Step 9: Create comprehensive comparison between theoretical and actual data
    print("\n=== PHASE 4: CREATING COMPARISON ANALYSIS ===")
    hourly_comparison_df = None
    try:
        hourly_comparison_df = create_complete_comparison(
            theoretical_detailed_df,
            theoretical_hourly_df,
            actual_minute_df,
            actual_hourly_df,
            inverter_capacity_kw=INVERTER_CAPACITY_KW,
        )

        if hourly_comparison_df is None:
            print("[INFO] No comparison DataFrame returned (hourly_comparison_df is None).")

        elif hourly_comparison_df.empty:
            print("[INFO] Comparison DataFrame is empty – likely no usable actual data "
                  "for this date range (e.g. today not finished yet).")

        else:
            print(f"[INFO] Comparison DataFrame has {len(hourly_comparison_df)} rows.")

            # Display Daily Averaged Soiling Losses
            print("\n=== DAILY AVERAGED SOILING LOSSES ===")

            col_name = "Daily Averaged Soiling Losses (%)"
            if col_name in hourly_comparison_df.columns:
                daily_losses = hourly_comparison_df[hourly_comparison_df[col_name].notna()]

                if not daily_losses.empty:
                    print(f"{'Date':<12} {col_name}")
                    print("=============================================")

                    for idx, row in daily_losses.iterrows():
                        date_str = idx.strftime("%Y-%m-%d")
                        loss_value = row[col_name]
                        print(f"{date_str:<12} {loss_value}")

                    # Push last day's value to HA
                    last_idx, last_row = list(daily_losses.tail(1).iterrows())[0]
                    last_date_str = last_idx.strftime("%Y-%m-%d")
                    last_loss_value = last_row[col_name]

                    update_daily_soiling_loss_to_ha(last_loss_value, last_date_str)

                else:
                    print("[INFO] No daily averaged soiling losses found (daily_losses is empty).")

            else:
                print(f"[INFO] Column '{col_name}' not found in hourly_comparison_df. "
                      "No soiling loss can be calculated.")

    except Exception as e:
        print(f"Error creating comparison: {e}")
        hourly_comparison_df = pd.DataFrame()

    
    # Step 10: Export all data to Excel
    print("\n=== PHASE 5: EXPORTING RESULTS TO EXCEL ===")
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
