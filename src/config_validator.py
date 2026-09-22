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

# Quick test if run directly
if __name__ == "__main__":
    validator = ConfigValidator("../config.yaml")
    config = validator.validate()
