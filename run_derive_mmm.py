"""
run_derive_mmm.py — Recover the TRUE NOAA CRW per-pixel MMM climatology.

WHY THIS EXISTS
---------------
Degree Heating Weeks (DHW) accumulate `HotSpot = SST - MMM`, but only on days
where HotSpot >= 1 degC. To compare OUR predicted DHW against NOAA's ground-truth
DHW fairly, both must use the SAME baseline climatology (the MMM).

NOAA defines MMM = per-pixel MAXIMUM of the 12 monthly-mean SSTs (a 1985-2012
climatology we do NOT have in our short 2017-2018 window). BUT the CRW 5km product
also publishes `CRW_HOTSPOT = SST - MMM` (raw, can be negative). Therefore:

        MMM = SST - HotSpot          (exact, per pixel, time-invariant)

We fetch a short window, verify MMM is static in time, take the temporal median
per pixel, interpolate it to our GBR training canvas, and save a tiny ~270 KB
sidecar (H x W float32). Storage-minimal: we never store full HotSpot time series.

Usage:
    python run_derive_mmm.py
    python run_derive_mmm.py --start 2018-07-01 --end 2018-07-31
"""
import argparse
import os

import numpy as np

from src.config_validator import ConfigValidator
from src.data_pipeline import DataIngestionFSM


def main():
    ap = argparse.ArgumentParser(description="Derive the true CRW MMM sidecar")
    ap.add_argument("--start", default="2018-07-01",
                    help="Snapshot window start (MMM is static, any window works)")
    ap.add_argument("--end", default="2018-07-31", help="Snapshot window end")
    args = ap.parse_args()

    cfg = ConfigValidator("config.yaml").validate()
    fsm = DataIngestionFSM(cfg)
    sst_src = cfg["data_sources"]["sst"]
    mirrors = sst_src.get("opendap_mirrors") or [sst_src["opendap_url"]]
    mirrors = list(dict.fromkeys(m for m in mirrors if m))

    # ── Fetch CRW_SST + CRW_HOTSPOT for a short window (try each mirror) ──
    ds = None
    for mirror in mirrors:
        try:
            ds = fsm.fetch_variables(mirror, ["CRW_SST", "CRW_HOTSPOT"],
                                     args.start, args.end)
            print(f"  Fetched MMM snapshot from {mirror.split('/')[2]}")
            break
        except Exception as e:
            print(f"  Mirror {mirror.split('/')[2]} failed: {str(e)[:80]}")
    if ds is None:
        raise SystemExit("Could not fetch CRW SST/HotSpot from any mirror.")

    # MMM = SST - HotSpot, per pixel. Verify it is time-invariant (sanity check).
    mmm_series = ds["CRW_SST"] - ds["CRW_HOTSPOT"]
    temporal_std = float(mmm_series.std("time").mean())
    print(f"  MMM temporal std (expect ~0): {temporal_std:.3e}")
    mmm_native = mmm_series.median("time")  # (lat, lon)

    # ── Interpolate to the exact training canvas grid ──
    canvas = cfg["spatial_canvas"]
    tlat = np.arange(canvas["lat_min"], canvas["lat_max"], canvas["target_resolution"])
    tlon = np.arange(canvas["lon_min"], canvas["lon_max"], canvas["target_resolution"])
    mmm = mmm_native.interp(lat=tlat, lon=tlon, method="linear",
                            kwargs={"fill_value": "extrapolate"})
    mmm = np.asarray(mmm.values, dtype=np.float32)
    frac_valid = float(np.isfinite(mmm).mean())
    print(f"  Canvas MMM: shape={mmm.shape} range={np.nanmin(mmm):.2f}..{np.nanmax(mmm):.2f} degC "
          f"({frac_valid*100:.1f}% valid pixels; rest is land)")

    # ── Save tiny sidecar ──
    out = cfg["storage"]["crw_mmm_path"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, lat=tlat.astype(np.float32),
                        lon=tlon.astype(np.float32), mmm=mmm)
    print(f"  Saved MMM sidecar: {out} ({os.path.getsize(out)/1e3:.1f} KB)")


if __name__ == "__main__":
    main()
