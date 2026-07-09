# Soiling Losses Calculations

## Overview

This document describes the methodology implemented in `Soiling_losses_Version3.py` to estimate soiling and shading losses on solar PV systems.

The system operates as a two-stage pipeline: a sunny day gate (`check_sunny_day.py`) runs first and determines whether conditions are suitable for analysis. Only if the day passes does `Soiling_losses_Version3.py` run the full soiling calculation.

---

## Data Pipeline & Sunny Day Gate

### Pipeline Overview

```
city_harvester.py          (runs after sunset)
      │
      │  saves weather_cache/{date}.json
      ▼
check_sunny_day.py         (reads cache, checks GHI ratio, updates HA)
      │
      │  if sunny → triggers automation
      ▼
Soiling_losses_Version3.py (soiling calculations only)
```

### Why data is pulled after sunset

`city_harvester.py` always runs after sunset, which means the OpenWeatherMap Solar API returns finalized data for the full day — not a morning forecast. This is critical: a morning forecast can predict clouds that never materialise, causing the gate to incorrectly block a usable day. By pulling after sunset, the `cloudy_sky.ghi` values reflect what actually happened.

### Sunny Day Criterion — per-site, via Open-Meteo (2026-07-09)

`check_sunny_day.py` first fetches the **querying account's own** `input_text.solar_system_latitude`/`longitude` from HA, then calls Open-Meteo's free archive API (`https://archive-api.open-meteo.com/v1/archive?latitude=...&longitude=...&start_date=...&end_date=...&hourly=cloud_cover&daily=sunrise,sunset&timezone=auto`) for that exact site — not a shared, one-size-fits-all cache.

```
avg(cloud_cover % during this site's own daylight hours) <= CLOUD_COVER_SUNNY_THRESHOLD_PCT (20%)
```

This replaced a real bug: `check_sunny_day.py` previously read only `weather_cache/{date}.json`, which `city_harvester.py` hardcodes to a single fixed location (Irbid, 32.5514/35.8515) — every account's sunny/cloudy verdict was being decided by Irbid's weather regardless of where that account's actual PV site is. Confirmed same-day availability empirically before shipping this: Open-Meteo's archive endpoint (ERA5-based reanalysis, 9–25 km resolution depending on model) does return full same-day hourly data despite the archive's documented multi-day finalization delay for the underlying reanalysis.

Cloud cover % is a different physical quantity than the old GHI energy ratio (thin cirrus vs. thick stratus attenuate very differently at the same coverage %), so the 20% threshold is its own calibration, not a port of the old 90% figure — worth revisiting with real data over time.

**No fallback**: if an account has no configured coordinates yet, or the Open-Meteo call fails for any reason, the day is deliberately treated as **CLOUDY** (`is_sunny = False`) rather than falling back to a different site's/metric's data — this is a fail-safe, not a fail-open. The old cache-based GHI ratio method (below) was removed entirely rather than kept as a fallback, since silently substituting a different location or a different metric on failure was exactly the kind of unverified assumption this fix was meant to eliminate.

Retired method, kept here only as historical reference for the ratio it used to compute:

```
total_cloudy_ghi / total_clear_ghi >= 0.9
```

Hours where `clear_sky_ghi < 50 W/m²` were excluded (nighttime and near-horizon).

### Separation of concerns

`check_sunny_day.py` and `Soiling_losses_Version3.py` are fully independent:

| Script | Responsibility |
|--------|---------------|
| `check_sunny_day.py` | Reads cached JSON, evaluates GHI ratio, updates HA sensors |
| `Soiling_losses_Version3.py` | Soiling calculations only — no sunny day logic |

`check_sunny_day.py` does **not** call the OpenWeatherMap API and no longer reads `weather_cache/{date}.json` at all. It calls Open-Meteo, keyed off this account's own coordinates; if those coordinates aren't configured or the call fails, it reports the day as cloudy rather than reading any fallback source.

---

## The Core Challenge

Soiling loss estimation without a physical reference baseline (no clean reference panel, no on-site pyranometer) must rely entirely on:

- Theoretical DC output modeled via pvlib + OpenWeatherMap irradiance data
- Actual DC output from the inverter
- Per-MPPT voltage and current readings from two MPPT trackers

Everything that cannot be measured directly must be inferred from the physics of what each signal represents.

---

## Oversized DC/AC Systems — The Clipping Problem

These solar systems have a DC/AC ratio greater than 1.0 (DC array capacity exceeds inverter AC capacity). At peak hours the inverter clips — the actual DC output is hard-capped at inverter capacity regardless of irradiance, soiling, or shading. Theoretical DC at those hours correctly models output above inverter capacity, so a naive comparison produces a false soiling signal.

### Example

| System | DC Array | Inverter AC | DC/AC Ratio |
|--------|----------|-------------|-------------|
| Typical oversized | 15 kWp | 10 kW | 1.5 |

On a clear summer midday, theoretical DC = 13 kW. Inverter clips at ~10 kW. Comparing 13 kW vs 10 kW gives a 23% apparent "loss" — none of which is soiling.

---

## Theoretical DC Capping

Rather than discarding clipping hours, the theoretical DC value is capped at `MAX_TOTAL_DC_POWER` — the actual peak DC output ever recorded by the inverter, pulled live from Home Assistant (`input_number.max_total_dc_power`, in Watts).

```python
max_dc_kw = MAX_TOTAL_DC_POWER / 1000
Theoretical DC Capped (kW) = min(Theoretical DC Output (kW), max_dc_kw)
```

During clipping hours both the capped theoretical and actual converge to the same ceiling, contributing ~0 to the daily loss calculation — no hours need to be discarded.

---

## Soiling Loss Calculation

### Method — Daily Energy Summation

Soiling is calculated once per day by comparing the sum of capped theoretical DC against the sum of actual DC across all valid hours.

**Step 1 — Determine valid hours**

An hour is excluded from the soiling calculation if and only if the MPPT current imbalance between the two trackers exceeds 5%. A high imbalance means one string is shading-contaminated, which would corrupt the soiling signal.

```python
excluded if: MPPT Current Difference (%) > 5%
```

If MPPT current data is missing for an hour, the hour is **included** — we cannot determine shading status, so we do not penalise it.

**Step 2 — Sum valid hours**

```python
theo_energy  = Σ Theoretical DC Capped (kW)  for all valid hours
actual_energy = Σ Actual DC Power (kW)        for all valid hours
```

**Step 3 — Compute daily soiling loss**

```python
soiling_loss = (theo_energy - actual_energy) / theo_energy × 100
soiling_loss = max(1.0, soiling_loss)   # minimum floor: always at least 1%
```

The 1% minimum floor reflects that some soiling is always present. If the model overestimates or actual slightly exceeds theoretical due to forecast error, the result is floored at 1% rather than reported as 0% or negative.

**Step 4 — Write result**

The daily soiling loss is written to the last timestamp of that day in the `Daily Soiling Loss (%)` column, and pushed to Home Assistant via `input_text.daily_soiling_loss`.

---

## Shading Loss Calculation

Shading is a localized loss — it hits one string more than the other. This asymmetry between the two MPPT trackers is the signal.

**Step 1 — Per-MPPT actual power from V×I sensors**

```python
MPPT1 Power (kW) = MPPT1_Voltage × MPPT1_Current / 1000
MPPT2 Power (kW) = MPPT2_Voltage × MPPT2_Current / 1000
```

**Step 2 — Per-MPPT theoretical (equal-string assumption)**

```python
P_theo_each = Theoretical DC Output (kW) / 2
```

**Step 3 — Loss per MPPT as % of its own theoretical**

```python
loss1_pct = ((P_theo_each - MPPT1_Power) / P_theo_each × 100).clip(lower=0)
loss2_pct = ((P_theo_each - MPPT2_Power) / P_theo_each × 100).clip(lower=0)
```

**Step 4 — Differential loss → shading candidate**

```python
differential_pct = |loss1_pct - loss2_pct|
Shading Loss (%) = differential_pct / 2     (clipped at 50%)
```

The differential is divided by 2 because shading hits only one of the two equal strings, so the differential equals twice the shading loss expressed as a fraction of total theoretical output.

**Valid hours for shading**

Shading is computed only within the shoulder window — hours where theoretical DC is between 15% and 75% of inverter capacity. This keeps the per-hour comparison clean by avoiding near-zero (noisy) and near-clipping hours. The shoulder window is used internally for shading only and does not affect the soiling calculation.

```python
shoulder_lower = INVERTER_CAPACITY_KW × 0.15
shoulder_upper = INVERTER_CAPACITY_KW × 0.75
```

**Daily shading average**

The daily shading figure is the mean of valid shoulder-window hours after 2-sigma outlier removal. A minimum of 3 valid hours is required to produce a daily figure.

---

## Excel Output Columns

The `Hourly Comparison` sheet contains exactly these columns:

| Column | Description |
|--------|-------------|
| `Theoretical DC Output (kW)` | Raw theoretical output from pvlib model (before cap) |
| `Theoretical DC Capped (kW)` | Theoretical capped at MAX_TOTAL_DC_POWER — used in soiling calculation |
| `Actual DC Power (kW)` | Actual DC output from inverter |
| `MPPT1 Voltage` | Raw voltage reading, tracker 1 |
| `MPPT2 Voltage` | Raw voltage reading, tracker 2 |
| `MPPT1 Current` | Raw current reading, tracker 1 |
| `MPPT2 Current` | Raw current reading, tracker 2 |
| `MPPT Current Difference (%)` | Imbalance between trackers — > 5% excludes the hour from soiling |
| `Shading Loss (%)` | Per-hour shading estimate (shoulder-window hours only) |
| `Daily Soiling Loss (%)` | Daily result — written on the last row of each day |
| `Daily Averaged Shading Loss (%)` | Daily shading average — written on the last row of each day |
| `date` | Calendar date |

---

## Limitations That Remain

1. **Model bias** — The theoretical model depends on OpenWeatherMap irradiance (1-hour resolution, grid-based). Irradiance forecast errors show up in the soiling figure and cannot be distinguished from real soiling on a single day. Tracking trends over multiple days reduces this noise.

2. **Equal string assumption** — `P_theo_each = theoretical_total / 2` assumes both strings are identical. If string lengths differ, this introduces a systematic offset in the per-MPPT loss calculation used for shading.

3. **Soiling vs degradation** — Long-term module degradation (~0.5%/year) also appears as a slow-growing energy gap. For day-to-day analysis it is negligible but becomes relevant on a multi-year timescale.

4. **No rain event detection** — After rainfall, soiling resets naturally. The current method has no mechanism to detect this and recalibrate. Adding precipitation data from the OpenWeatherMap response would allow rain-event flagging and baseline resets.

---

## Cross-Site GHI Reuse: DNI/DHI Must Be Re-Derived, Not Copied (2026-07-09)

When `fetch_solar_forecast()` reuses a cached `weather_cache/{date}.json` (distance to the cache's own `coordinates` < `CACHE_DISTANCE_THRESHOLD_KM`), it now treats only `ghi` as regionally transferable. `dni`/`dhi` are **not** copied from the cache — they depend on solar zenith angle, which is specific to whichever site actually measured/queried that GHI, not to the site running this analysis.

`recompute_dni_dhi_for_site()` re-derives `dni`/`dhi` per hourly interval via pvlib's Erbs decomposition, using **this account's own** `LATITUDE`/`LONGITUDE`/`ALTITUDE` (already fetched from HA earlier in `main()`) to compute solar position — the same Erbs pattern `local_solar_harvester.py` already uses for the sensor's own site, just re-anchored to whichever site is actually running the calculation. This applies uniformly, whether the cache came from `city_harvester.py` (OpenWeatherMap, fixed Irbid coordinates) or from the local GHI sensor harvester (a specific customer site's coordinates) — DNI/DHI are always recomputed for the querying site, never inherited.

`CACHE_DISTANCE_THRESHOLD_KM` (30 km) was left unchanged — this fix removes the correctness risk of sharing DNI/DHI, but doesn't itself justify widening how far GHI is borrowed from.

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SHADING_EXCLUSION_MPPT_DIFF` | 5.0% | MPPT current imbalance threshold — hours above this are excluded from soiling |
| `QUALITY_MIN_MPPT_CURRENT` | 0.1 A | Min current per MPPT for V×I power calculation to be valid |
| `SHOULDER_LOWER_RATIO` | 0.15 | Min theoretical DC as fraction of inverter capacity (shading window only) |
| `SHOULDER_UPPER_RATIO` | 0.75 | Max theoretical DC as fraction of inverter capacity (shading window only) |
| `MIN_DAILY_DATA_POINTS` | 3 | Min shoulder-window hours required to produce a daily shading figure |
| `TEMP_COEFFICIENT` | -0.0035 | Power temperature coefficient per °C (verify against module datasheet) |
| `CACHE_DISTANCE_THRESHOLD_KM` | 30 | Max distance (km) between site and cached weather location to reuse GHI (DNI/DHI are always re-derived for the querying site, never reused as-is) |
| `CLOUD_COVER_SUNNY_THRESHOLD_PCT` (`check_sunny_day.py`) | 20% | Avg. daylight-hour cloud cover at/below which a site's own day is considered sunny (Open-Meteo path) |
