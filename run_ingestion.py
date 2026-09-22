from src.config_validator import ConfigValidator
from src.data_pipeline import DataIngestionFSM

def main():
    print("Initializing Data Ingestion Pipeline...")
    
    # Validate the config.yaml at the project root
    validator = ConfigValidator("config.yaml")
    valid_config = validator.validate()
    
    # Run the FSM pipeline
    pipeline = DataIngestionFSM(valid_config)
    pipeline.run()

if __name__ == "__main__":
    main()
