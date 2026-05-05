from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIR_EDA = PROJECT_ROOT/"reports"/"figures"/"EDA_file"
DIR_ENG = PROJECT_ROOT/"reports"/"figures"/"Feature_Engineering_file"
DIR_MODEL = PROJECT_ROOT/"reports"/"figures"/"Modeling_file"
LOAD_PROCESSED_DATA = PROJECT_ROOT/"data"/"processed"
LOAD_RAW_DATA = PROJECT_ROOT/"data"/"raw"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
