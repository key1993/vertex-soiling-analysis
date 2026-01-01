import requests
import json
import sys
from datetime import datetime

# Get parameters from command line
HA_URL = sys.argv[1]
HA_TOKEN = sys. argv[2]
FORECAST_DATE = sys.argv[3]

# Get latitude/longitude from HA
def get_coordinates_from_ha():
    headers = {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}
    
    lat_url = f"{HA_URL}/api/states/input_text.solar_system_latitude"
    lon_url = f"{HA_URL}/api/states/input_text.solar_system_longitude"
    
    lat = float(requests.get(lat_url, headers=headers).json().get('state', 0))
    lon = float(requests.get(lon_url, headers=headers).json().get('state', 0))
    
    return lat, lon

# Fetch solar forecast
def fetch_solar_forecast(lat, lon, date, api_key="4d9f599d094608656284561fba4a79f7"):
    url = (
        f"https://api.openweathermap.org/energy/2.0/solar/interval_data"
        f"?lat={lat}&lon={lon}&date={date}&interval=1h&appid={api_key}"
    )
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"ERROR: Solar API request failed:  {response.status_code}")
        sys.exit(1)

# Check if sunny - UPDATED THRESHOLD TO 0.9
def is_sunny_day(solar_data, threshold=0.9, min_sunny_hours=6):
    """
    Determine if day is sunny enough for soiling analysis. 
    
    threshold=0.9 means cloudy_sky_ghi must be at least 90% of clear_sky_ghi
    This is VERY STRICT - only accepts nearly perfect clear days
    """
    sunny_count = 0
    total_daylight = 0
    cloud_ratios = []
    
    for interval in solar_data. get('intervals', []):
        try:
            clear_ghi = interval['avg_irradiance']['clear_sky']['ghi']
            cloudy_ghi = interval['avg_irradiance']['cloudy_sky']['ghi']
            
            if clear_ghi < 50:   # Skip night hours
                continue
            
            total_daylight += 1
            cloud_ratio = cloudy_ghi / clear_ghi if clear_ghi > 0 else 0
            cloud_ratios.append(cloud_ratio)
            
            if cloud_ratio >= threshold:
                sunny_count += 1
                
        except Exception as e:
            continue
    
    if total_daylight > 0:
        sunshine_pct = (sunny_count / total_daylight) * 100
        avg_cloud = sum(cloud_ratios) / len(cloud_ratios) if cloud_ratios else 0
    else:
        sunshine_pct = 0
        avg_cloud = 0
    
    is_sunny = sunny_count >= min_sunny_hours
    
    return {
        'is_sunny': is_sunny,
        'sunny_hours': sunny_count,
        'total_daylight_hours': total_daylight,
        'sunshine_percentage': round(sunshine_pct, 1),
        'avg_cloud_ratio': round(avg_cloud, 3)
    }

# Update Home Assistant sensors
def update_ha_sensors(result, date_str):
    headers = {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}
    
    # Update boolean
    boolean_url = f"{HA_URL}/api/services/input_boolean/turn_{'on' if result['is_sunny'] else 'off'}"
    requests.post(boolean_url, headers=headers, json={"entity_id": "input_boolean.sunny_day_detected"})
    
    # Update text info with threshold information
    text_url = f"{HA_URL}/api/services/input_text/set_value"
    info_text = f"{result['sunshine_percentage']}% sunny ({result['sunny_hours']}/{result['total_daylight_hours']}h @ 90% threshold) on {date_str}"
    requests.post(text_url, headers=headers, json={
        "entity_id": "input_text.sunny_day_info",
        "value": info_text
    })
    
    print(f"✅ Updated HA:  {'SUNNY' if result['is_sunny'] else 'CLOUDY'} - {info_text}")

# Main execution
if __name__ == "__main__":
    print(f"🌤️ Checking if {FORECAST_DATE} is sunny (STRICT:  90% threshold)...")
    
    lat, lon = get_coordinates_from_ha()
    print(f"📍 Location: {lat}, {lon}")
    
    solar_data = fetch_solar_forecast(lat, lon, FORECAST_DATE)
    
    # ⭐ THRESHOLD SET TO 0.9 (90% clear sky required)
    result = is_sunny_day(solar_data, threshold=0.9, min_sunny_hours=6)
    
    print(f"☀️ Sunny Hours: {result['sunny_hours']}/{result['total_daylight_hours']}")
    print(f"📊 Sunshine:  {result['sunshine_percentage']}%")
    print(f"🌥️ Avg Cloud Ratio: {result['avg_cloud_ratio']}")
    
    update_ha_sensors(result, FORECAST_DATE)
    
    # Exit with code 0 if sunny, 1 if cloudy (for GitHub Actions conditional)
    if result['is_sunny']:
        print("✅ Day is SUNNY (≥90% clear) - proceeding with analysis")
        sys.exit(0)
    else:
        print("❌ Day is CLOUDY (<90% clear) - skipping analysis")
        sys.exit(1)
