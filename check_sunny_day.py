import requests
import sys
import time

# Get parameters from command line
HA_URL = sys.argv[1]
HA_TOKEN = sys.argv[2]
FORECAST_DATE = sys.argv[3]
ACCOUNT_ID = sys.argv[4] if len(sys.argv) >= 5 else ""

def _api(path):
    """Build a full Ebsher API URL scoped to the current account."""
    sep = '&' if '?' in path else '?'
    return f"{HA_URL}{path}{sep}account_id={ACCOUNT_ID}"

# Average cloud cover (%) across this site's own daylight hours, at or below
# which the day is considered sunny enough for soiling analysis. Cloud cover %
# is not the same metric as an energy ratio (thin cirrus vs. thick stratus
# attenuate irradiance very differently at the same coverage %), so this
# threshold is its own calibration.
CLOUD_COVER_SUNNY_THRESHOLD_PCT = 20.0

HA_COORD_SENSORS = {
    "lat": "input_text.solar_system_latitude",
    "lon": "input_text.solar_system_longitude",
}

# Multiple accounts' automations can dispatch within seconds of each other and
# briefly overload the backend, making an individual request time out even
# though the backend is otherwise healthy - retry a couple of times before
# treating this like an actual "not configured" gap.
COORD_FETCH_RETRIES = 3
COORD_FETCH_RETRY_DELAY_SEC = 3

# 1. Validate and clean HA_URL
if not HA_URL.startswith(('http://', 'https://')):
    print("❌ ERROR: Invalid Home Assistant URL format!")
    sys.exit(1)

HA_URL = HA_URL.rstrip('/')


def fetch_site_coordinates():
    """Fetch this account's own solar_system_latitude/longitude from HA.
    Returns (lat, lon, failure_reason) - failure_reason is None on success,
    otherwise a short string distinguishing a real config gap ("not
    configured") from a transient backend problem ("timed out", "HTTP 503",
    ...) so a burst of overlapping requests doesn't get silently lumped in
    with an account that genuinely has no coordinates set."""
    headers = {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}
    coords = {}
    for key, entity_id in HA_COORD_SENSORS.items():
        last_error = None
        for attempt in range(1, COORD_FETCH_RETRIES + 1):
            try:
                resp = requests.get(_api(f"/api/states/{entity_id}"), headers=headers, timeout=10)

                if resp.status_code >= 500:
                    # Likely transient (e.g. backend overloaded by a burst of
                    # simultaneous account dispatches) - worth retrying.
                    raise requests.exceptions.RequestException(f"HTTP {resp.status_code}")

                if resp.status_code != 200:
                    print(f"[WARNING] {entity_id} returned HTTP {resp.status_code} - not retrying")
                    return None, None, f"HTTP {resp.status_code} fetching {entity_id}"

                value = resp.json().get("state")
                if value in (None, "unknown", "unavailable", ""):
                    print(f"[WARNING] {entity_id} state is '{value}' - not configured yet")
                    return None, None, "not configured"

                coords[key] = float(value)
                break

            except requests.exceptions.Timeout:
                last_error = "timed out after 10s"
            except requests.exceptions.RequestException as e:
                last_error = str(e)

            if attempt < COORD_FETCH_RETRIES:
                print(f"[WARNING] Attempt {attempt}/{COORD_FETCH_RETRIES} to fetch {entity_id} failed ({last_error}) - retrying in {COORD_FETCH_RETRY_DELAY_SEC}s...")
                time.sleep(COORD_FETCH_RETRY_DELAY_SEC)
            else:
                print(f"[WARNING] Could not fetch {entity_id} after {COORD_FETCH_RETRIES} attempts: {last_error}")
                return None, None, f"backend unreachable ({last_error})"

    return coords.get("lat"), coords.get("lon"), None


def fetch_cloud_cover_for_site(lat, lon, date):
    """Fetch this specific site's own hourly cloud cover + sunrise/sunset from
    Open-Meteo's archive API (free, no API key, parameterized per lat/lon -
    unlike the shared single-city weather_cache/{date}.json)."""
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}&start_date={date}&end_date={date}"
        "&hourly=cloud_cover&daily=sunrise,sunset&timezone=auto"
    )
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    hourly_time = data["hourly"]["time"]
    hourly_cloud = data["hourly"]["cloud_cover"]
    sunrise = data["daily"]["sunrise"][0]
    sunset = data["daily"]["sunset"][0]

    # timezone=auto returns local-time ISO strings with no UTC offset suffix,
    # so lexical comparison against sunrise/sunset (same format, same day) is safe.
    daylight_values = [
        cloud for ts, cloud in zip(hourly_time, hourly_cloud)
        if sunrise <= ts <= sunset and cloud is not None
    ]

    if not daylight_values:
        raise ValueError("No daylight-hour cloud cover values returned")

    avg_cloud_cover = sum(daylight_values) / len(daylight_values)

    return {
        'is_sunny': avg_cloud_cover <= CLOUD_COVER_SUNNY_THRESHOLD_PCT,
        'avg_cloud_cover_pct': round(avg_cloud_cover, 1),
        'usable_daylight_hours': len(daylight_values),
        'sunrise': sunrise,
        'sunset': sunset,
    }


def update_ha_sensors(is_sunny, info_text):
    headers = {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}

    boolean_url = _api(f"/api/services/input_boolean/turn_{'on' if is_sunny else 'off'}")
    requests.post(boolean_url, headers=headers, json={"entity_id": "input_boolean.sunny_day_detected"})

    text_url = _api("/api/services/input_text/set_value")
    requests.post(text_url, headers=headers, json={
        "entity_id": "input_text.sunny_day_info",
        "value": info_text
    })

    print(f"✅ HA Updated: {info_text}")


# Main execution
if __name__ == "__main__":
    print(f"Analyzing {FORECAST_DATE} for this account's own site...")

    lat, lon, coord_failure_reason = fetch_site_coordinates()
    result = None

    if lat is not None and lon is not None:
        try:
            result = fetch_cloud_cover_for_site(lat, lon, FORECAST_DATE)
            print(f"📊 Site coordinates: {lat}, {lon}")
            print(f"☁️ Avg daylight cloud cover: {result['avg_cloud_cover_pct']}%")
        except Exception as e:
            coord_failure_reason = f"Open-Meteo check failed: {e}"

    if result is None:
        # No coordinates configured, or Open-Meteo/backend unreachable - fail
        # safe to "cloudy" rather than guessing from an unrelated location or
        # a different metric, so the soiling calculation never runs on a day
        # whose sky conditions could not actually be verified for this site.
        reason = coord_failure_reason or "unknown"
        print(f"[WARNING] Cannot verify sky conditions for this site ({reason}) - treating {FORECAST_DATE} as CLOUDY")
        result = {'is_sunny': False, 'avg_cloud_cover_pct': None, 'usable_daylight_hours': 0, 'reason': reason}

    status_label = "SUNNY" if result['is_sunny'] else "CLOUDY"
    if result['avg_cloud_cover_pct'] is not None:
        info_text = (
            f"{status_label}: {result['avg_cloud_cover_pct']}% avg cloud cover "
            f"({result['usable_daylight_hours']}h daylight, site {lat:.4f},{lon:.4f}) on {FORECAST_DATE}"
        )
    else:
        info_text = f"{status_label}: unable to verify sky conditions for this site on {FORECAST_DATE} ({result['reason']})"

    update_ha_sensors(result['is_sunny'], info_text)

    # Exit codes for GitHub Actions / Automation
    if result['is_sunny']:
        print("✅ SUCCESS: Site is sunny enough for soiling analysis.")
        sys.exit(0)
    else:
        print("❌ FAILED: Site is too cloudy (or unverifiable) for soiling analysis.")
        sys.exit(1)
