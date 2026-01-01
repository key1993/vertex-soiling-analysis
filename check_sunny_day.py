import requests
import json
import sys
from datetime import datetime

# ==========================================================
# CONFIGURATION & PARAMETERS
# ==========================================================
try:
    HA_URL = sys.argv[1].rstrip('/') # Remove trailing slash if present
    HA_TOKEN = sys.argv[2]
    FORECAST_DATE = sys.argv[3]
except IndexError:
    print("❌ ERROR: Missing arguments.")
    print("Usage: python script.py <HA_URL> <HA_TOKEN> <YYYY-MM-DD>")
    sys.exit(1)

# API Key for OpenWeatherMap
OWM_API_KEY = "4d9f599d094608656284561fba4a79f7"

def get_coordinates_from_ha():
    """Fetches latitude and longitude from Home Assistant input_text entities."""
    headers = {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json"
    }
    
    lat_url = f"{HA_URL}/api/states/input_text.solar_system_latitude"
    lon_url = f"{HA_URL}/api/states/input_text.solar_system_longitude"
    
    try:
        lat_res = requests.get(lat_url, headers=headers, timeout=10)
        lon_res = requests.get(lon_url, headers=headers, timeout=10)
        
        lat_res.raise_for_status()
        lon_res.raise_for_status()
        
        lat = float(lat_res.json().get('state', 0))
        lon = float(lon_res.json().get('state', 0))
        
        return lat, lon
    except Exception as e:
        print(f"❌ ERROR: Failed to get coordinates from HA: {e}")
        sys.exit(1)

def fetch_solar_forecast(lat, lon, date):
    """Fetches solar irradiance data from OpenWeatherMap."""
    url = (
        f"https://api.openweathermap.org/energy/2.0/solar/interval_data"
        f"?lat={lat}&lon={lon}&date={date}&interval=1h&appid={OWM_API_KEY}"
    )
    
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ ERROR: Solar API request failed: {response.status_code}")
            print(f"Response: {response.text}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: Connection to Solar API failed: {e}")
        sys.exit(1)

def is_sunny_day(solar_data, threshold=0.9, min_sunny_hours=6):
    """
    Analyzes hourly data. 
    A 'sunny' hour must have Cloudy GHI >= 90% of Clear Sky GHI.
    """
    sunny_count = 0
    total_daylight = 0
    cloud_ratios = []
    
    intervals = solar_data.get('intervals', [])
    
    for interval in intervals:
        try:
            # Irradiance values (Global Horizontal Irradiance)
            clear_ghi = interval['avg_irradiance']['clear_sky']['ghi']
            cloudy_ghi = interval['avg_irradiance']['cloudy_sky']['ghi']
            
            # Skip hours with very low sun (night/dawn/dusk)
            if clear_ghi < 50:
                continue
            
            total_daylight += 1
            ratio = cloudy_ghi / clear_ghi if clear_ghi > 0 else 0
            cloud_ratios.append(ratio)
            
            if ratio >= threshold:
                sunny_count += 1
                
        except (KeyError, TypeError):
            continue
    
    # Calculate statistics
    sunshine_pct = (sunny_count / total_daylight * 100) if total_daylight > 0 else 0
    avg_cloud = sum(cloud_ratios) / len(cloud_ratios) if cloud_ratios else 0
    
    # Final Verdict
    is_sunny = sunny_count >= min_sunny_hours
    
    return {
        'is_sunny': is_sunny,
        'sunny_hours': sunny_count,
        'total_daylight_hours': total_daylight,
        'sunshine_percentage': round(sunshine_pct, 1),
        'avg_cloud_ratio': round(avg_cloud, 3)
    }

def update_ha_sensors(result, date_str):
    """Sends the analysis results back to Home Assistant."""
    headers = {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # 1. Update the Binary Status (input_boolean)
    state_service = "turn_on" if result['is_sunny'] else "turn_off"
    boolean_url = f"{HA_URL}/api/services/input_boolean/{state_service}"
    requests.post(boolean_url, headers=headers, json={"entity_id": "input_boolean.sunny_day_detected"})
    
    # 2. Update the Information Text (input_text)
    text_url = f"{HA_URL}/api/services/input_text/set_value"
    info_text = (
        f"{result['sunshine_percentage']}% sunny "
        f"({result['sunny_hours']}/{result['total_daylight_hours']}h @ 90% threshold) on {date_str}"
    )
    requests.post(text_url, headers=headers, json={
        "entity_id": "input_text.sunny_day_info",
        "value": info_text[:255] # Ensure it stays within HA character limits
    })
    
    print(f"✅ Updated HA: {'SUNNY' if result['is_sunny'] else 'CLOUDY'}")
    print(f"📊 Info: {info_text}")

# ==========================================================
# MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    print(f"🌤️ Checking if {FORECAST_DATE} is sunny (STRICT: 90% threshold)...")
    
    # 1. Get Lat/Lon
    lat, lon = get_coordinates_from_ha()
    print(f"📍 Location: {lat}, {lon}")
    
    # 2. Get Weather Data
    solar_data = fetch_solar_forecast(lat, lon, FORECAST_DATE)
    
    # 3. Analyze Data
    result = is_sunny_day(solar_data, threshold=0.9, min_sunny_hours=6)
    
    # 4. Report to HA
    update_ha_sensors(result, FORECAST_DATE)
    
    # 5. Exit logic for automation pipelines
    if result['is_sunny']:
        print("✅ Day is SUNNY - Proceeding.")
        sys.exit(0)
    else:
        print("❌ Day is CLOUDY - Skipping.")
        sys.exit(1)
