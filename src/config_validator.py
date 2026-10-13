import yaml
import datetime

# It's important to validate the configuration before running heavy data operations.
# This class checks that spatial and temporal bounds make logical sense
# (e.g., start date is before end date, latitudes are within valid ranges).

class ConfigValidator:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
            
    def validate(self):
        """Runs all checks and returns the validated configuration dictionary."""
        print(f"Validating pipeline configuration: {self.config['pipeline']['name']}")
        self._check_spatial_bounds()
        self._check_temporal_bounds()
        self._check_strategies()
        self._check_pipeline_consistency()  # pipeline block mirrors canvases + sane params
        self._check_coverage()  # requested canvas vs common (minimal) / maximal (envelope)
        print("Validation Successful! All parameters are sensible.\n")
        return self.config

    def _check_spatial_bounds(self):
        space = self.config["spatial_canvas"]
        
        # Latitude must be between -90 (South Pole) and 90 (North Pole)
        assert -90 <= space["lat_min"] <= 90, "lat_min must be between -90 and 90"
        assert -90 <= space["lat_max"] <= 90, "lat_max must be between -90 and 90"
        
        # The bottom of our box (min) must be below the top (max)
        assert space["lat_min"] < space["lat_max"], "lat_min must be strictly less than lat_max"
        
        # Longitude must be between -180 and 180 (or 0 and 360 depending on convention, we assume standard -180 to 180 here)
        # Note: GBR is around 142 to 154, so it falls in standard bounds.
        assert -180 <= space["lon_min"] <= 180 or 0 <= space["lon_min"] <= 360, "lon_min out of logical bounds"
        assert space["lon_min"] < space["lon_max"], "lon_min must be strictly less than lon_max"

    def _check_temporal_bounds(self):
        time = self.config["temporal_canvas"]
        
        # Convert the string dates from the yaml (e.g., "2020-01-01") into Python Date objects
        start = datetime.datetime.strptime(time["start_date"], "%Y-%m-%d")
        end = datetime.datetime.strptime(time["end_date"], "%Y-%m-%d")
        
        # Time must flow forward!
        assert start < end, "Start date must be strictly before end date!"

    def _check_strategies(self):
        time = self.config["temporal_canvas"]

        valid_cycles = ["daily", "weekly", "monthly", "annually"]
        assert time["time_lag_cycle"] in valid_cycles, f"Lag cycle must be one of {valid_cycles}"

        valid_strats = ["min", "max", "avg"]
        assert time["lag_strategy"] in valid_strats, f"Lag strategy must be one of {valid_strats}"

    @staticmethod
    def _parse_date(s):
        import datetime as _dt
        if s in (None, "present"):
            return _dt.date.today()
        return _dt.datetime.strptime(s, "%Y-%m-%d").date()

    def _check_pipeline_consistency(self):
        """Pipeline block is the control panel — it must agree with canvases."""
        pipe = self.config.get("pipeline", {})
        if not pipe or "time_start" not in pipe:
            return  # legacy config without expanded pipeline block
        space, tcanvas = self.config["spatial_canvas"], self.config["temporal_canvas"]
        assert pipe["time_start"] == tcanvas["start_date"], \
            f"pipeline.time_start {pipe['time_start']} != temporal_canvas.start_date {tcanvas['start_date']}"
        assert pipe["time_end"] == tcanvas["end_date"], \
            f"pipeline.time_end {pipe['time_end']} != temporal_canvas.end_date {tcanvas['end_date']}"
        assert pipe.get("time_lag_cycle") == tcanvas.get("time_lag_cycle"), \
            "pipeline.time_lag_cycle must match temporal_canvas.time_lag_cycle"
        assert pipe.get("lag_strategy") == tcanvas.get("lag_strategy"), \
            "pipeline.lag_strategy must match temporal_canvas.lag_strategy"
        for k in ("lat_min", "lat_max", "lon_min", "lon_max", "target_resolution"):
            assert pipe[k] == space[k], f"pipeline.{k} {pipe[k]} != spatial_canvas.{k} {space[k]}"
        assert pipe.get("strict_real_only") is True and pipe.get("allow_synthetic") is False, \
            "pipeline must be strict real-only (strict_real_only=true, allow_synthetic=false)"
        ing = pipe.get("ingestion", {})
        assert ing.get("chunk_years", 1) >= 1, "ingestion.chunk_years must be >= 1"
        assert ing.get("max_retries", 1) >= 1, "ingestion.max_retries must be >= 1"
        assert ing.get("fetch_order", ["erddap_nc"])[:1] == ["erddap_nc"] or \
            set(ing.get("fetch_order", [])) <= {"erddap_nc", "pydap"}, \
            "ingestion.fetch_order must be subset of [erddap_nc, pydap]"
        val = pipe.get("validation", {})
        assert 0 <= val.get("max_nan_fraction", 0) <= 1, "validation.max_nan_fraction in [0,1]"
        assert val.get("sst_min", -99) < val.get("sst_max", 99), "sst_min < sst_max"

    def _check_coverage(self):
        """Enforce: requested canvas ⊆ maximal (hard), ⊆ common (strict warning).

        common_coverage  = minimal intersection, all vars real (velocity-limited).
        maximal_coverage = widest envelope, SST-only extension allowed.
        """
        if "common_coverage" not in self.config or "maximal_coverage" not in self.config:
            return  # legacy config without coverage blocks
        space, tcanvas = self.config["spatial_canvas"], self.config["temporal_canvas"]
        common, maximal = self.config["common_coverage"], self.config["maximal_coverage"]

        req_start = self._parse_date(tcanvas["start_date"])
        req_end = self._parse_date(tcanvas["end_date"])
        max_start = self._parse_date(maximal["start_date"])
        max_end = self._parse_date(maximal["end_date"])
        assert max_start <= req_start < req_end <= max_end, (
            f"Requested [{req_start},{req_end}] outside maximal "
            f"[{max_start},{max_end}]. No source covers that range."
        )
        assert (maximal["lat_min"] <= space["lat_min"] < space["lat_max"] <= maximal["lat_max"]
                and maximal["lon_min"] <= space["lon_min"] < space["lon_max"] <= maximal["lon_max"]), \
            "Requested spatial canvas outside maximal_coverage."

        com_start = self._parse_date(common["start_date"])
        com_end = self._parse_date(common["end_date"])
        inside = (com_start <= req_start and req_end <= com_end
                  and common["lat_min"] <= space["lat_min"]
                  and space["lat_max"] <= common["lat_max"]
                  and common["lon_min"] <= space["lon_min"]
                  and space["lon_max"] <= common["lon_max"])
        if not inside:
            print(f"  !! WARNING: requested canvas outside common_coverage "
                  f"[{com_start},{com_end}]. Velocity missing there by design "
                  f"(SST-only extension). Strict PINN runs should stay inside common.")

# Quick test if run directly
if __name__ == "__main__":
    validator = ConfigValidator("../config.yaml")
    config = validator.validate()
