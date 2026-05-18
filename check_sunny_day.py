import requests
import json
import sys
import os
from datetime import datetime

# Get parameters from command line
HA_URL = sys.argv[1]
HA_TOKEN = sys.argv[2]
FORECAST_DATE = sys.argv[3]
ACCOUNT_ID = sys.argv[4] if len(sys.argv) >= 5 else ""

def _api(path):
    """Build a full Ebsher API URL scoped to the current account."""
    sep = '&' if '?' in path else '?'
    return f"{HA_URL}{path}{sep}account_id={ACCOUNT_ID}"

CACHE_DIR = "weather_cache"

# 1. Validate and clean HA_URL
if not HA_URL.startswith(('http://', 'https://')):
    print("❌ ERROR: Invalid Home Assistant URL format!")
    sys.exit(1)

HA_URL = HA_URL.rstrip('/')

# 2. Load solar data from the cached JSON file produced by city_harvester.py
def load_solar_from_cache(date):
    cache_file = os.path.join(CACHE_DIR, f"{date}.json")
    if not os.path.exists(cache_file):
        print(f"❌ ERROR: Cache file not found: {cache_file}")
        print(f"   Run city_harvester.py first to collect data for {date}.")
        sys.exit(1)
    with open(cache_file, 'r') as f:
        cached = json.load(f)
    solar = cached.get("solar_forecast")
    if not solar:
        print(f"❌ ERROR: No 'solar_forecast' key found in {cache_file}")
        sys.exit(1)
    print(f"[OK] Loaded solar data from cache: {cache_file}")
    return solar

# 3. Determine if day is sunny (Based on Total Daily Energy)
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
    boolean_url = _api(f"/api/services/input_boolean/turn_{'on' if result['is_sunny'] else 'off'}")
    requests.post(boolean_url, headers=headers, json={"entity_id": "input_boolean.sunny_day_detected"})

    # Update text info
    text_url = _api("/api/services/input_text/set_value")
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
    print(f"Analyzing {FORECAST_DATE} based on Total Daily Energy (Threshold: 90%)...")

    solar_data = load_solar_from_cache(FORECAST_DATE)

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
