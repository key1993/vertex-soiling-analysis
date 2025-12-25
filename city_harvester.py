import requests
import json
import os
from datetime import datetime, timedelta, timezone

# === CONFIGURATION ===
API_KEY = "4d9f599d094608656284561fba4a79f7"  #
CACHE_DIR = "weather_cache"
CITY_NAME = "Irbid"
LAT = 32.5514
LON = 35.8515

def harvest_daily_data():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        
    today_str = datetime.now().strftime("%Y-%m-%d")
    print(f"[INFO] Starting harvest for {CITY_NAME} on {today_str}...")

    # 1. Fetch Solar Irradiance Data (Energy API)
    solar_url = (
        f"https://api.openweathermap.org/energy/2.0/solar/interval_data"
        f"?lat={LAT}&lon={LON}&date={today_str}&interval=1h&appid={API_KEY}"
    )
    
    # 2. Fetch Ambient Weather Data (OneCall TimeMachine)
    # We fetch data for each of the 24 hours of 'today'
    weather_hourly = []
    base_ts = int(datetime.strptime(today_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    
    for h in range(24):
        ts = base_ts + (h * 3600)
        url = f"https://api.openweathermap.org/data/3.0/onecall/timemachine?lat={LAT}&lon={LON}&dt={ts}&appid={API_KEY}&units=metric"
        try:
            resp = requests.get(url).json()
            data = resp.get("data", [{}])[0]
            weather_hourly.append({
                "hour": h,
                "temp": data.get("temp"),
                "wind_speed": data.get("wind_speed")
            })
        except Exception as e:
            print(f"[ERROR] Hour {h} weather failed: {e}")

    # 3. Consolidate into one JSON file per day
    try:
        solar_resp = requests.get(solar_url).json()
        
        final_package = {
            "date": today_str,
            "city": CITY_NAME,
            "coordinates": {"lat": LAT, "lon": LON},
            "solar_forecast": solar_resp,
            "ambient_weather": weather_hourly,
            "harvested_at": datetime.now().isoformat()
        }
        
        file_path = os.path.join(CACHE_DIR, f"{today_str}.json")
        with open(file_path, "w") as f:
            json.dump(final_package, f, indent=4)
            
        print(f"[OKEY] Daily data saved to {file_path}")
        
    except Exception as e:
        print(f"[ERROR] Harvest failed: {e}")

if __name__ == "__main__":
    harvest_daily_data()
