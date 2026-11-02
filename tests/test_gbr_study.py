"""Scientific contracts for the common window, wave fields and sparse cohort."""
from datetime import date
import json

import numpy as np
import polars as pl
import pytest
import xarray as xr

from src.reef.catalog import SOURCES
from src.reef.colonies import audit_colony_overlap, prepare_colonies
from src.reef.harmonize import harmonize
from src.reef.planning import GBR_BOUNDS, Job, load_config, source_names
from src.reef.region import geometry


def test_study_windows_and_geography():
    west, south, east, north = geometry().bounds
    assert GBR_BOUNDS["lat_min"] <= south < north <= GBR_BOUNDS["lat_max"]
    assert GBR_BOUNDS["lon_min"] <= west < east <= GBR_BOUNDS["lon_max"]
    for profile, start, end, fields in [
            ("gbr_wave_long", "1993-01-01", "2026-08-31", 36),
            ("gbr_all_daily", "2010-12-01", "2019-04-30", 142)]:
        config = load_config("configs/gbr_study.yaml", profile, common_window=True)
        assert all(config["grid"][k] == v for k, v in GBR_BOUNDS.items())
        assert config["time"] == {"start": start, "end": end}
        assert sum(len(SOURCES[n]["features"]) for n in source_names(config)) == fields
    with pytest.raises(ValueError, match="explicit dates"):
        load_config("configs/research.yaml", overrides={"time": {"end": end}}, common_window=True)


def test_wave_daily_mean_requires_exact_slots_and_shared_daily_index():
    times = np.arange(np.datetime64("2026-08-31T00"), np.datetime64("2026-09-01T00"), np.timedelta64(3, "h"))
    coords = {"time": times, "lat": [-18.25, -18, -17.75], "lon": [146, 146.25, 146.5]}
    values = np.broadcast_to(np.arange(8)[:, None, None], (8, 3, 3)).astype(float)
    wave = xr.Dataset({f.native: (("time", "lat", "lon"), values.copy(), {"units": f.units})
                       for f in SOURCES["waves"]["features"]}, coords=coords)
    job = Job("t", "2026-08-31", "2026-08-31", (-18.125, -17.875), (146.125, 146.375), .25)
    daily = harmonize(wave, "waves", job)
    assert daily.time.dtype == "datetime64[ns]"
    np.testing.assert_allclose(daily.to_array(), 3.5)
    wave["VHM0"].values[7, :, :] = np.nan
    assert np.isnan(harmonize(wave, "waves", job).significant_wave_height.item())
    for invalid in [wave.isel(time=slice(1, None)), wave.assign_coords(time=times + np.timedelta64(1, "m"))]:
        with pytest.raises(ValueError, match="UTC slots"):
            harmonize(invalid, "waves", job)


def test_colony_scores_and_history_units(tmp_path, monkeypatch):
    raw = tmp_path / "fixture.csv"
    raw.write_text("lat,long,Bleaching,Date_2022,Date_bleaching\n"
                   "-18.8259,147.62727,1,1/01/2022,28/03/2022\n"
                   "-18.8259,147.62727,6,1/01/2022,28/03/2022\n"
                   "-18.8259,147.62727,4,1/01/2022,27/03/2022\n"
                   "0,0,3,1/01/2022,28/03/2022\n")
    labels = prepare_colonies(raw)
    assert labels["severity_eligible"].to_list() == [True, False, True]
    observations = tmp_path / "observations.parquet"
    labels.write_parquet(observations)
    day = date(2022, 3, 19)
    grid = dict(lat_min=-19, lat_max=-18.75, lon_min=147.5, lon_max=147.75, resolution=.25)
    plan = {"cadence": "daily", "feature_names": ["x"], "grid": grid,
            "requested": {"start": str(day), "end": str(day)}}
    (tmp_path / "manifest.json").write_text(json.dumps({"fingerprint": "fixture", "plan": plan}))
    cube = xr.Dataset({"x": (("time", "lat", "lon"), [[[2.0]]]), "model_domain": (("lat", "lon"), [[True]])},
                      coords={"lat": [-18.875], "lon": [147.625], "time": [np.datetime64(day)]})
    monkeypatch.setattr("src.reef.colonies.open_cube", lambda root: cube)
    result = audit_colony_overlap(tmp_path, str(observations), lookback=1, lead=7)
    assert result["distinct_cells"] == result["distinct_histories"] == 1
    assert result["row_counts"]["matched"] == 2 and result["row_counts"]["incomplete_history"] == 1
    assert result["histories"][0]["ordinal_rows"] == 1
    cube["model_domain"].values[:] = False
    assert audit_colony_overlap(tmp_path, observations, 1, 7)["row_counts"]["outside_domain"] == 3


def test_colour_chart_stream_stays_local_and_bounded(tmp_path):
    import csv
    import io
    import time
    from src.reef.coralwatch import BoundedReader, COLUMNS, subset_surveys
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=[*COLUMNS, "Submitted by"])
    writer.writeheader()
    row = {"Activity ID": "s", "Observation date": "2026-03-01T14:00:00Z", "Latitude": "-18.8259",
           "Longitude": "147.62727", "Average": "4.5", "Submitted by": "excluded"}
    writer.writerow(row)
    writer.writerow({**row, "Latitude": "0"})
    writer.writerow({**row, "Average": "7"})
    stream.seek(0)
    file = tmp_path / "surveys.parquet"
    assert subset_surveys(stream, file) == 3
    result = pl.read_parquet(file)
    assert result["valid_chart_range"].to_list() == [True, False]
    assert result["local_survey_date"].to_list() == [date(2026, 3, 2)] * 2
    assert "Submitted by" not in result.columns
    limited = BoundedReader(io.BytesIO(b"1234"), 3, time.monotonic() + 10)
    with pytest.raises(ValueError, match="budget"):
        limited.readinto(bytearray(8))


def test_survey_denominator_and_block_rotation(tmp_path):
    from datetime import timedelta
    from src.reef.study import blocked_folds, chart_surveys
    file = tmp_path / "chart.parquet"
    rows = [{"survey_id": "A", "lat": -18.2, "lon": 147, "local_survey_date": date(2025,4,1),
             "coral_type": "Plate corals", "coral_number": str(i), "valid_chart_range": True,
             "lightest": "D2" if i < 6 else "D4", "darkest": "D2" if i < 6 else "D4",
             "chart_mean": 2 if i < 6 else 4} for i in range(20)]
    pl.DataFrame(rows + [{**rows[0], "coral_type": "Soft corals"}]).write_parquet(file)
    survey = chart_surveys(file).to_dicts()[0]
    assert survey["qc"] and survey["chart_flag"] and survey["hard_corals"] == 20
    pl.DataFrame(rows + [rows[-1]]).write_parquet(file)
    assert not chart_surveys(file)["qc"][0]
    data = pl.DataFrame({"cell_lat": [-18.0]*5, "cell_lon": [146., 145., 146.25, 148., 148.],
                         "bucket": [0,1,2,2,0], "local_survey_date": [
                             date(2025,3,1),date(2022,3,1),date(2010,3,1),date(2010,3,1),date(2025,1,3)]})
    folds = blocked_folds(data, 0, timedelta(days=22), 50)
    assert [len(folds[k]) for k in ("train","validation","test")] == [1,1,1]
    assert folds["train"]["cell_lon"][0] == 148.0
