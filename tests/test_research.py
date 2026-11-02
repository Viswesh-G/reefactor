"""Contracts and fault injection; generated fixtures never enter production data."""
import copy
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from src.reef.catalog import SOURCES
from src.reef.features import derive_features
from src.reef.harmonize import area_mean, harmonize, validate_partition
from src.reef.observability import exclusive_run, summarize
from src.reef.pipeline import StateMachine, open_cube, run_pipeline
from src.reef.planning import Job, fingerprint, load_config, make_plan, validate_config


def configuration(tmp_path):
    cfg = load_config("configs/research.yaml")
    cfg["storage"]["root"] = str(tmp_path)
    cfg["grid"]["tile_cells"] = 2
    return cfg


def fixture_fetch(job, cfg, names, scratch):
    lat, lon = job.centers()
    times = np.arange(np.datetime64(job.start), np.datetime64(job.end)+1)
    shape = (len(times), len(lat), len(lon))
    return xr.Dataset({"sea_surface_temperature": (("time", "lat", "lon"), np.full(shape, 28, dtype="float32")),
                       "degree_heating_weeks": (("time", "lat", "lon"), np.zeros(shape, dtype="float32"))},
                      coords={"time": times, "lat": lat, "lon": lon}), [{"test_fixture": True}]


def test_plan_and_resolution_guard(tmp_path):
    cfg = configuration(tmp_path)
    assert make_plan(cfg)["jobs"] == 4
    cfg["profile"] = "daily"
    assert make_plan(cfg)["fields"] == 30
    cfg["grid"]["resolution"] = 0.05
    with pytest.raises(ValueError, match="upsampling"):
        validate_config(cfg)


def test_fingerprint_tracks_versions(tmp_path):
    cfg = configuration(tmp_path)
    before = fingerprint(cfg)
    cfg["versions"]["noaa"] = "changed"
    assert fingerprint(cfg) != before


def test_conservative_constant_and_no_extrapolation():
    field = xr.DataArray(np.full((4, 4), 7.0), dims=("lat", "lon"),
                         coords={"lat": [-0.375, -0.125, 0.125, 0.375], "lon": [0.125, 0.375, 0.625, 0.875]})
    np.testing.assert_allclose(area_mean(field, np.array([-0.25, 0.25]), np.array([0.25, 0.75]), 0.5), 7)
    with pytest.raises(ValueError, match="extrapolation"):
        area_mean(field, np.array([-0.75, -0.25]), np.array([0.25, 0.75]), 0.5)


def test_land_unknown_support_and_gaps_are_masked(tmp_path):
    cfg = configuration(tmp_path)
    job = Job("x", "2018-01-01", "2018-01-03", (-18, -17), (146, 147), 0.25)
    ds, _ = fixture_fetch(job, cfg, ["noaa"], tmp_path)
    ds.sea_surface_temperature.values[:, 0, 0] = np.nan
    checked, metrics = validate_partition(ds, ["noaa"], 0.05)
    assert not checked.common_valid.values[:, 0, 0].any()
    assert metrics["sea_surface_temperature"]["supported_cell_fraction"] < 1
    ds.sea_surface_temperature.values[1] = np.nan
    with pytest.raises(ValueError, match="intermittent"):
        validate_partition(ds, ["noaa"], 0.05)


def test_fsm_rejects_invalid_transition():
    machine = StateMachine("INIT", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="Illegal"):
        machine.move("READY")


def test_end_to_end_resume_and_corruption(tmp_path):
    cfg = configuration(tmp_path)
    root = run_pipeline(cfg, fixture_fetch, lambda *a: None)
    with open_cube(root) as cube:
        assert dict(cube.sizes) == {"time": 3, "lat": 4, "lon": 4}
        assert float(cube.sea_surface_temperature.mean().compute()) == 28
    assert len(list((root/"tiles").glob("*.zarr"))) == 4
    def no_network(*a):
        raise AssertionError("Resume fetched a committed partition")
    run_pipeline(cfg, no_network, lambda *a: None)
    assert summarize(root)["storage"][0]["partitions"] == 4
    target = next((root/"tiles").glob("*.zarr"))
    (target/".zattrs").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="integrity"):
        run_pipeline(cfg, no_network, lambda *a: None)


def test_failure_releases_lock_and_preserves_commits(tmp_path):
    cfg = configuration(tmp_path)
    cfg["execution"]["workers"] = 1
    calls = 0
    def flaky(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ConnectionError("injected network failure")
        return fixture_fetch(*args)
    with pytest.raises(ConnectionError):
        run_pipeline(cfg, flaky, lambda *a: None)
    root = Path(cfg["storage"]["root"])/fingerprint(cfg)[:16]
    assert len(list((root/"tiles").glob("*.zarr"))) == 1
    assert json.loads((root/"manifest.json").read_text())["status"] == "ERROR"
    run_pipeline(cfg, fixture_fetch, lambda *a: None)
    assert json.loads((root/"manifest.json").read_text())["status"] == "READY"


def test_output_lock_excludes_another_writer(tmp_path):
    with exclusive_run(tmp_path):
        with pytest.raises(RuntimeError, match="Another process"):
            with exclusive_run(tmp_path):
                pass


def test_bounded_concurrency(tmp_path):
    cfg = configuration(tmp_path)
    active, maximum = 0, 0
    lock = threading.Lock()
    def measured(*args):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.03)
        result = fixture_fetch(*args)
        with lock:
            active -= 1
        return result
    run_pipeline(cfg, measured, lambda *a: None)
    assert 1 < maximum <= cfg["execution"]["workers"]


def test_dhw_needs_full_history_and_counts_full_hotspot():
    ds = xr.Dataset({"sea_surface_temperature": (("time", "lat", "lon"), np.full((85, 1, 1), 32))},
                    coords={"time": np.arange("2018-01-01", "2018-03-27", dtype="datetime64[D]"), "lat": [0], "lon": [0]})
    mmm = xr.DataArray([[30]], dims=("lat", "lon"), coords={"lat": [0], "lon": [0]})
    result = derive_features(ds, mmm)
    assert np.isnan(result.derived_dhw.values[:83]).all()
    assert result.derived_dhw.values[-1, 0, 0] == 24


def test_heat_budget_manufactured_solution_and_gradients():
    torch = pytest.importorskip("torch")
    from src.reef.physics import ForcedSurfaceBudget
    rho, cp, h, q = 1025, 3990, 20, 100
    increment = q*86400/(rho*cp*h)
    temperatures = torch.stack([torch.full((1, 5, 5), 28.0, dtype=torch.float64),
                                torch.full((1, 5, 5), 28+increment, dtype=torch.float64)], dim=1).requires_grad_()
    shape = (1, 1, 5, 5)
    forcing = {"water_u": torch.zeros(shape), "water_v": torch.zeros(shape),
               "mixed_layer_depth": torch.full(shape, h), "net_heat_flux": torch.full(shape, q)}
    criterion = ForcedSurfaceBudget(learn_k=True)
    loss = criterion(temperatures, forcing, [-18, -17.75, -17.5, -17.25, -17], [146, 146.25, 146.5, 146.75, 147])
    assert float(loss) < 1e-16
    loss.backward()
    assert torch.isfinite(temperatures.grad).all()


def test_multivariate_forecaster_shape_and_backward():
    torch = pytest.importorskip("torch")
    from src.reef.learning import MultivariateSSTForecaster
    torch.set_num_threads(1)
    model = MultivariateSSTForecaster(feature_count=3, hidden_channels=4, horizon=2)
    result = model(torch.randn(2, 3, 6, 5, 5))
    assert result.shape == (2, 2, 5, 5)
    result.square().mean().backward()
    assert model.encoder.conv.weight.grad is not None


def test_ram_cache_is_bounded_and_single_flight():
    from concurrent.futures import ThreadPoolExecutor
    from src.reef.cache import ByteCache
    cache = ByteCache(10)
    calls = 0
    lock = threading.Lock()
    def loader():
        nonlocal calls
        with lock:
            calls += 1
        time.sleep(0.05)
        return b"123456"
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(lambda _: cache.get("same", loader), range(4)))
    assert calls == 1 and all(value == b"123456" for value, hit in outputs)
    cache.get("other", lambda: b"abcdef")
    assert cache.bytes <= 10 and cache.peak_bytes <= 10
    with pytest.raises(ValueError):
        cache.get("failed", lambda: (_ for _ in ()).throw(ValueError("injected")))
    assert cache.get("failed", lambda: b"ok")[0] == b"ok"


def test_hourly_flux_time_shift_and_complete_days():
    from src.reef.catalog import Feature
    name = "test_hourly"
    SOURCES[name] = {"cadence": "hourly", "kind": "test_fixture", "resolution": 0.25,
                     "features": [Feature("shortwave_net", "mean_test_flux", "W m-2")]}
    try:
        times = np.arange("2018-01-01T00", "2018-01-03T01", dtype="datetime64[h]")
        raw = np.broadcast_to(np.arange(len(times))[:, None, None], (len(times), 4, 4)).astype(float).copy()
        ds = xr.Dataset({"mean_test_flux": (("time", "lat", "lon"), raw, {"units": "W m**-2"})},
                        coords={"time": times, "lat": [-0.375, -0.125, 0.125, 0.375],
                                "lon": [0.125, 0.375, 0.625, 0.875]})
        job = Job("x", "2018-01-01", "2018-01-02", (-0.5, 0.5), (0, 1), 0.5)
        result = harmonize(ds, name, job)
        np.testing.assert_allclose(result.shortwave_net.values[:, 0, 0], [12.5, 36.5])
        incomplete = harmonize(ds.isel(time=slice(None, -1)), name, job)
        assert np.isnan(incomplete.shortwave_net.values[1]).all()
    finally:
        SOURCES.pop(name)


def test_monthly_context_is_not_daily_and_is_causally_labeled():
    name = "test_monthly"
    from src.reef.catalog import Feature
    SOURCES[name] = {"cadence": "monthly", "kind": "test_fixture", "resolution": 0.25,
                    "features": [Feature("ph", "ph", "1")]}
    try:
        ds = xr.Dataset({"ph": (("time", "lat", "lon"), np.full((1, 4, 4), 8), {"units": "1"})},
             coords={"time": np.array(["2018-01-16"], dtype="datetime64[D]"),
                     "lat": [-0.375, -0.125, 0.125, 0.375], "lon": [0.125, 0.375, 0.625, 0.875]})
        result = harmonize(ds, name, Job("x", "2018-01-01", "2018-01-31", (-0.5, 0.5), (0, 1), 0.5))
        assert result.sizes["time"] == 1
        assert result.available_not_before.values[0] == np.datetime64("2018-02-01")
    finally:
        SOURCES.pop(name)


def test_salinity_budget_freshwater_dilution():
    torch = pytest.importorskip("torch")
    from src.reef.physics import ForcedSurfaceBudget
    freshwater, rho, depth, initial = 0.001, 1025, 20, 35
    change = -initial*freshwater*86400/(rho*depth)
    salt = torch.stack([torch.full((1, 5, 5), initial, dtype=torch.float64),
                        torch.full((1, 5, 5), initial+change, dtype=torch.float64)], dim=1)
    shape = (1, 1, 5, 5)
    forcing = {"water_u": torch.zeros(shape), "water_v": torch.zeros(shape),
               "mixed_layer_depth": torch.full(shape, depth),
               "freshwater_flux": torch.full(shape, freshwater, dtype=torch.float64)}
    loss = ForcedSurfaceBudget()(salt, forcing, [-18, -17.75, -17.5, -17.25, -17],
                                [146, 146.25, 146.5, 146.75, 147], kind="salt")
    assert float(loss) < 1e-16


def test_label_extraction_and_past_only_sampling(tmp_path):
    from src.reef.labels import CONTEXT, IDENTIFIERS, convert_labels, sample_history
    import polars as pl
    row = {name: "example" for name in IDENTIFIERS}
    row.update({name: "1" for name in CONTEXT})
    row.update(Date="2018-01-05", Latitude_Degrees="0.2", Longitude_Degrees="0.2", Percent_Bleaching="25")
    path = tmp_path/"labels.csv"
    pl.DataFrame([row]).write_csv(path)
    summary = convert_labels(path, tmp_path/"labels.parquet")
    assert summary["valid_targets"] == 1
    cube = xr.Dataset({"sst": (("time", "lat", "lon"), np.ones((6, 2, 2)))},
        coords={"time": np.arange("2018-01-01", "2018-01-07", dtype="datetime64[D]"), "lat": [0.125, 0.375], "lon": [0.125, 0.375]})
    history = sample_history(cube, lat=0.2, lon=0.2, survey_date="2018-01-05", features=["sst"], lookback=2, lead_days=2)
    assert history.time.values[-1] == np.datetime64("2018-01-03")
    with pytest.raises(ValueError, match="outside"):
        sample_history(cube, lat=2, lon=2, survey_date="2018-01-05", features=["sst"])


def test_remote_subset_never_builds_a_global_dask_graph(monkeypatch):
    """A virtual 64 GB source must fit a tiny selected window without expanding it."""
    import dask.array as da
    import zarr
    from src.reef.catalog import Feature
    from src.reef import remote_zarr
    store = zarr.storage.MemoryStore()
    group = zarr.group(store=store)
    for name, values, units in [("latitude", np.arange(128)*0.5-32, "degrees_north"),
                                ("longitude", np.arange(256)*0.5, "degrees_east"),
                                ("elevation", -np.arange(50)-0.5, "m"),
                                ("time", np.arange(10000), "days since 2018-01-01")]:
        coordinate = group.create_dataset(name, data=values, fill_value=None)
        coordinate.attrs.update(_ARRAY_DIMENSIONS=[name], units=units)
    array = group.create_dataset("thetao", shape=(10000, 50, 128, 256),
                                  chunks=(2000, 1, 16, 16), dtype="f4", fill_value=np.nan)
    array.attrs.update(_ARRAY_DIMENSIONS=["time", "elevation", "latitude", "longitude"], units="degrees_C")
    array[:3, 0, 63:69, :5] = 28
    zarr.consolidate_metadata(store)
    store.url, store.bytes, store.requests, store.attempts = "fixture", 0, 0, 0
    store.cache_hits, store.cache_bytes, store.digests, store.started = 0, 0, {}, time.monotonic()
    monkeypatch.setattr(remote_zarr, "HTTPZarrStore", lambda *a, **kw: store)
    original = da.from_array
    def bounded_graph(array, *args, **kwargs):
        if getattr(array, "size", 0) > 1_000_000:
            raise AssertionError("Global source was converted to Dask before selection")
        return original(array, *args, **kwargs)
    monkeypatch.setattr(da, "from_array", bounded_graph)
    source = {"features": [Feature("ocean_temperature", "thetao", "degree_Celsius")],
              "zarr_urls": ["fixture"], "zarr_version": "test", "dataset_id": "fixture",
              "product": "fixture", "resolution": 0.5, "cadence": "daily", "kind": "test_fixture",
              "lat_bounds": [-32, 31.5]}
    SOURCES["test_huge"] = source
    try:
        cfg = load_config("configs/research.yaml")
        result, _ = remote_zarr.fetch_public_zarr(source, "test_huge",
                  Job("x", "2018-01-01", "2018-01-03", (0, 1), (0, 1), 0.5), cfg)
        assert result.ocean_temperature.shape == (3, 2, 2)
        np.testing.assert_allclose(result.ocean_temperature, 28)
    finally:
        SOURCES.pop("test_huge")


def test_fixed_domain_masks_every_feature_in_every_month(tmp_path):
    cfg = configuration(tmp_path)
    cfg["time"]["end"] = "2018-02-03"
    cfg["grid"]["tile_cells"] = 4
    def gap_in_one_feature(job, *args):
        ds, details = fixture_fetch(job, *args)
        if job.start.startswith("2018-02"):
            ds.sea_surface_temperature.values[0, 0, 0] = np.nan
        return ds, details
    root = run_pipeline(cfg, gap_in_one_feature, lambda *a: None)
    with open_cube(root) as cube:
        assert not bool(cube.model_domain.values[0, 0])
        assert np.isnan(cube.degree_heating_weeks.values[:, 0, 0]).all()
        assert np.isnan(cube.sea_surface_temperature.values[:, 0, 0]).all()
        np.testing.assert_array_equal(np.isfinite(cube.degree_heating_weeks), np.isfinite(cube.sea_surface_temperature))
    with open_cube(root, model_ready=False) as raw:
        assert np.isfinite(raw.degree_heating_weeks.values[:, 0, 0]).all()


def test_missing_feature_tile_and_coordinate_drift_are_rejected(tmp_path):
    cfg = configuration(tmp_path)
    def missing(*args):
        ds, details = fixture_fetch(*args)
        return ds.drop_vars("degree_heating_weeks"), details
    with pytest.raises(ValueError, match="missing required features"):
        run_pipeline(cfg, missing, lambda *a: None)
    def shifted(*args):
        ds, details = fixture_fetch(*args)
        return ds.assign_coords(lon=ds.lon+0.01), details
    with pytest.raises(ValueError, match="common coordinates"):
        run_pipeline(cfg, shifted, lambda *a: None)


def test_other_regions_are_rejected(tmp_path):
    cfg = configuration(tmp_path)
    cfg["grid"].update(lat_min=0, lat_max=1)
    with pytest.raises(ValueError, match="GBR study"):
        validate_config(cfg)


def test_weight_budgets_keep_all_130_features_and_both_heads():
    torch = pytest.importorskip("torch")
    from src.reef.learning import build_forecaster, parameter_count
    torch.set_num_threads(1)
    for budget in (1000, 10000, 100000, 1000000):
        model = build_forecaster(130, budget, horizon=2)
        assert budget*0.95 <= parameter_count(model) <= budget
        assert model.project.in_channels == 260
    model = build_forecaster(130, 1000, horizon=2)
    inputs = torch.randn(1, 3, 260, 3, 3)
    result = model(inputs, return_bleaching=True)
    assert result["sst"].shape == (1, 2, 3, 3)
    assert result["bleaching_percent"].shape == (1, 3, 3)
    (result["sst"].square().mean()+result["bleaching_percent"].mean()).backward()
    assert bool((model.project.weight.grad.abs().sum(dim=(0, 2, 3)) > 0).all())


def test_opendap_transport_keeps_byte_guards_and_provenance(monkeypatch):
    import requests
    from pydap.model import DatasetType, BaseType
    from pydap.handlers.lib import BaseHandler
    from webob import Request
    from src.reef.catalog import Feature
    from src.reef.opendap import fetch_opendap
    dataset = DatasetType("fixture")
    for name, data, units in [("time", np.arange(3, dtype=float), "days since 2018-01-01"),
                             ("latitude", np.arange(6)*0.25-18.125, "degrees_north"),
                             ("longitude", np.arange(6)*0.25+145.875, "degrees_east")]:
        dataset[name] = BaseType(name, data=data, dims=(name,), attributes={"units": units})
    dataset["temp"] = BaseType("temp", data=np.full((3, 6, 6), 28, dtype="float32"),
        dims=("time", "latitude", "longitude"), attributes={"units": "degree_Celsius"})
    handler, calls = BaseHandler(dataset), []
    def local_response(session, prepared, **kwargs):
        from urllib.parse import urlsplit
        url = urlsplit(prepared.url)
        response = Request.blank(url.path+("?"+url.query if url.query else "")).get_response(handler)
        out = requests.Response()
        out.status_code, out._content, out._content_consumed = response.status_int, response.body, True
        out.headers.update(response.headers)
        out.url, out.request = prepared.url, prepared
        calls.append(prepared.url)
        return out
    monkeypatch.setattr(requests.Session, "send", local_response)
    source = {"features": [Feature("ereefs_fixture", "temp", "degree_Celsius")],
        "url": "https://fixture.test/data", "product": "fixture", "version": "fixture",
        "resolution": 0.25, "lat_bounds": [-90, 90], "cadence": "daily", "kind": "test_fixture",
        "temporal_statistic": "daily_snapshot"}
    SOURCES["test_transport"] = source
    try:
        cfg = load_config("configs/research.yaml")
        output, provenance = fetch_opendap(source, "test_transport",
            Job("x", "2018-01-01", "2018-01-03", (-18, -17), (146, 147), 0.25), cfg)
        np.testing.assert_allclose(output.ereefs_fixture, 28)
        assert provenance["requests"] == len(calls) and provenance["requests"] >= 4
        assert provenance["download_bytes"] > 0
        cfg["execution"]["max_download_mb"] = 0.000001
        with pytest.raises(RuntimeError, match="transfer/time budget"):
            fetch_opendap(source, "test_transport",
                Job("x", "2018-01-01", "2018-01-03", (-18, -17), (146, 147), 0.25), cfg)
    finally:
        SOURCES.pop("test_transport")


def test_native_memory_budget_shrinks_rich_tiles():
    from src.reef.planning import effective_tile_cells
    cfg = load_config("configs/research.yaml", "gbr_rich_public", {"time": {"end": "2018-01-31"}})
    assert effective_tile_cells(cfg) < cfg["grid"]["tile_cells"]
    cfg["execution"]["max_tile_mb"] = 0.000001
    with pytest.raises(ValueError, match="one target cell"):
        effective_tile_cells(cfg)


def test_sparse_biological_targets_do_not_create_negative_labels(tmp_path):
    torch = pytest.importorskip("torch")
    import polars as pl
    from datetime import date
    from src.reef.learning import FeatureWindowDataset
    cfg = configuration(tmp_path/"run")
    cfg["grid"]["tile_cells"] = 4
    cfg["time"]["end"] = "2018-02-09"
    root = run_pipeline(cfg, fixture_fetch, lambda *a: None)
    with open_cube(root) as cube:
        row, col = np.argwhere(cube.model_domain.values)[0]
        lat, lon = float(cube.lat[row]), float(cube.lon[col])
    labels = tmp_path/"sparse.parquet"
    pl.DataFrame({"survey_date": [date(2018, 1, 5)], "lat": [lat], "lon": [lon],
                  "bleaching_percent": [25.0], "target_valid": [True]}).write_parquet(labels)
    ds = FeatureWindowDataset(root, ["sea_surface_temperature"], lookback=2, horizon=2,
                              patch_size=4, labels_path=labels)
    try:
        assert ds[0]["bleaching_target"].numel() == 0
        assert ds[1]["bleaching_target"].tolist() == [25.0]
        assert ds[1]["bleaching_sites"].shape == (1, 2)
    finally:
        ds.close()


def test_geographic_tile_pruning_keeps_one_complete_full_gbr_schema(tmp_path):
    cfg = configuration(tmp_path)
    cfg["grid"].update(lat_min=-24.0, lat_max=-10.0, lon_min=142.0, lon_max=154.0, tile_cells=16)
    root = run_pipeline(cfg, fixture_fetch, lambda *a: None)
    with open_cube(root) as cube:
        assert cube.sizes["lat"] == 56 and cube.sizes["lon"] == 48
        assert np.isnan(cube.sea_surface_temperature.values[:, ~cube.model_domain.values]).all()
        np.testing.assert_array_equal(np.isfinite(cube.sea_surface_temperature), np.isfinite(cube.degree_heating_weeks))
