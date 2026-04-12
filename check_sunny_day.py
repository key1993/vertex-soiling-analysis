import requests
import json
import sys
import os
from datetime import datetime

# Get parameters from command line
HA_URL = sys.argv[1]
HA_TOKEN = sys.argv[2]
FORECAST_DATE = sys.argv[3]

OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")
if not OPENWEATHER_API_KEY:
    print("❌ ERROR: OPENWEATHER_API_KEY environment variable is not set.")
    sys.exit(1)

# 1. Validate and clean HA_URL
if not HA_URL.startswith(('http://', 'https://')):
    print("❌ ERROR: Invalid Home Assistant URL format!")
    sys.exit(1)

HA_URL = HA_URL.rstrip('/')

# 2. Get latitude/longitude from HA
def get_coordinates_from_ha():
    headers = {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}
    
    lat_url = f"{HA_URL}/api/states/input_text.solar_system_latitude"
    lon_url = f"{HA_URL}/api/states/input_text.solar_system_longitude"
    
    try:
        lat = float(requests.get(lat_url, headers=headers).json().get('state', 0))
        lon = float(requests.get(lon_url, headers=headers).json().get('state', 0))
        return lat, lon
    except Exception as e:
        print(f"❌ ERROR: Failed to fetch coordinates from HA: {e}")
        sys.exit(1)

# 3. Fetch solar forecast
def fetch_solar_forecast(lat, lon, date, api_key=None):
    if api_key is None:
        api_key = OPENWEATHER_API_KEY
    url = (
        f"https://api.openweathermap.org/energy/2.0/solar/interval_data"
        f"?lat={lat}&lon={lon}&date={date}&interval=1h&appid={api_key}"
    )
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"❌ ERROR: Solar API request failed: {response.status_code}")
        sys.exit(1)

# 4. Determine if day is sunny (Based on Total Daily Energy)
def calculate_daily_sunniness(solar_data, threshold=0.9):
    """
    Calculates the ratio of total actual energy vs total potential energy.
    This ignores 'how many hours' and looks at the total volume of light.
    """
    total_clear_ghi = 0
    total_cloudy_ghi = 0
    usable_hours = 0
    
    for interval in solar_data.get('intervals', []):
        try:
            clear_ghi = interval['avg_irradiance']['clear_sky']['ghi']
            cloudy_ghi = interval['avg_irradiance']['cloudy_sky']['ghi']
            
            # Skip low-light hours (Sunrise/Sunset/Night) to keep data clean
            if clear_ghi < 50:
                continue
            
            total_clear_ghi += clear_ghi
            total_cloudy_ghi += cloudy_ghi
            usable_hours += 1
                
        except (KeyError, TypeError):
            continue
    
    if total_clear_ghi > 0:
        actual_ratio = total_cloudy_ghi / total_clear_ghi
        sunshine_pct = actual_ratio * 100
    else:
        actual_ratio = 0
        sunshine_pct = 0
    
    # Day is 'Sunny' only if the total energy is >= 90% of theoretical max
    is_sunny = actual_ratio >= threshold
    
    return {
        'is_sunny': is_sunny,
        'sunshine_percentage': round(sunshine_pct, 1),
        'total_clear_energy': round(total_clear_ghi, 2),
        'total_cloudy_energy': round(total_cloudy_ghi, 2),
        'usable_daylight_hours': usable_hours
    }

# 5. Update Home Assistant sensors
def update_ha_sensors(result, date_str):
    headers = {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}
    
    # Update boolean (On/Off)
    boolean_url = f"{HA_URL}/api/services/input_boolean/turn_{'on' if result['is_sunny'] else 'off'}"
    requests.post(boolean_url, headers=headers, json={"entity_id": "input_boolean.sunny_day_detected"})
    
    # Update text info
    text_url = f"{HA_URL}/api/services/input_text/set_value"
    status_label = "SUNNY" if result['is_sunny'] else "CLOUDY"
    info_text = (
        f"{status_label}: {result['sunshine_percentage']}% total sunlight potential "
        f"({result['usable_daylight_hours']}h analyzed) on {date_str}"
    )
    
    requests.post(text_url, headers=headers, json={
        "entity_id": "input_text.sunny_day_info",
        "value": info_text
    })
    
    print(f"✅ HA Updated: {info_text}")

# Main execution
if __name__ == "__main__":
    print(f"🌤️ Analyzing {FORECAST_DATE} based on Total Daily Energy (Threshold: 90%)...")
    
    lat, lon = get_coordinates_from_ha()
    print(f"📍 Location: {lat}, {lon}")
    
    solar_data = fetch_solar_forecast(lat, lon, FORECAST_DATE)
    
    # Run the new energy-volume logic
    result = calculate_daily_sunniness(solar_data, threshold=0.9)
    
    print(f"📊 Total Clear-Sky GHI: {result['total_clear_energy']}")
    print(f"📊 Total Actual GHI:    {result['total_cloudy_energy']}")
    print(f"☀️ Final Ratio:        {result['sunshine_percentage']}%")
    
    update_ha_sensors(result, FORECAST_DATE)
    
    # Exit codes for GitHub Actions / Automation
    if result['is_sunny']:
        print("✅ SUCCESS: Total energy meets 90% threshold.")
        sys.exit(0)
    else:
        print("❌ FAILED: Total energy below 90% threshold.")
        sys.exit(1)
