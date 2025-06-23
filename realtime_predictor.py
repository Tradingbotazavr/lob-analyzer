import pandas as pd
import joblib
import os
import time
import glob
from loguru import logger

# --- Configuration ---
MODEL_PATH = "models/random_forest_model.pkl"
DATA_DIR = "data/merged/"
LOG_FILE = "logs/realtime_predictions.log"

# How often to check for new data
LOOP_INTERVAL_SECONDS = 5
# How far back to look for data points in each iteration
DATA_WINDOW_SECONDS = 10
# How many rows to read from the end of the file (as a buffer)
TAIL_ROWS_TO_READ = 100 

# --- Logging Setup ---
logger.add(LOG_FILE, rotation="10 MB", compression="zip", level="INFO")


# --- Core Functions ---

def load_model(model_path: str):
    """Loads model artifacts from a file."""
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return None, None
    try:
        artifacts = joblib.load(model_path)
        model = artifacts["model"]
        features = artifacts["feature_names"]
        logger.success(f"Successfully loaded model and {len(features)} features from {model_path}")
        return model, features
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        return None, None

def find_latest_file(directory: str) -> str | None:
    """Finds the most recently modified file in a directory."""
    list_of_files = glob.glob(os.path.join(directory, '*.parquet'))
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getmtime)

def predict(df: pd.DataFrame, model, feature_names: list[str]) -> pd.Series:
    """Generates predictions for a given DataFrame."""
    # Ensure all required features are present
    if not all(col in df.columns for col in feature_names):
        missing = list(set(feature_names) - set(df.columns))
        logger.error(f"Missing features in input data: {missing}")
        return pd.Series(dtype=int)

    # Use only the required features and handle potential NaNs
    X = df[feature_names].dropna()
    if X.empty:
        return pd.Series(dtype=int)

    predictions = model.predict(X)
    return pd.Series(predictions, index=X.index, name="prediction")


# --- Main Loop ---

def run_predictor():
    """Main real-time prediction loop."""
    logger.info("Starting real-time predictor...")
    model, features = load_model(MODEL_PATH)

    if model is None:
        logger.critical("Could not load model. Exiting.")
        return

    # Keep track of the timestamp of the last processed row
    last_processed_ts = 0

    while True:
        try:
            latest_file = find_latest_file(DATA_DIR)
            if not latest_file:
                logger.warning(f"No data files in {DATA_DIR}. Waiting...")
                time.sleep(LOOP_INTERVAL_SECONDS)
                continue

            # This is not the most efficient way for huge files, but it's robust
            # for the file sizes expected in this project.
            df = pd.read_parquet(latest_file)
            
            # Filter for new rows since the last check
            new_rows_df = df[df['ts'] > last_processed_ts].copy()

            if not new_rows_df.empty:
                logger.info(f"Found {len(new_rows_df)} new data point(s) in '{os.path.basename(latest_file)}'.")
                
                predictions = predict(new_rows_df, model, features)
                
                if not predictions.empty:
                    results = new_rows_df.join(predictions)
                    
                    # Log the results for rows that have a prediction
                    output_cols = ["ts", "imbalance", "price", "prediction"]
                    display_cols = [col for col in output_cols if col in results.columns]
                    
                    for _, row in results.dropna(subset=['prediction']).iterrows():
                        log_items = [f"{col}={row[col]}" for col in display_cols]
                        logger.success(f"PREDICTION >> {', '.join(log_items)}")
                
                # Update the timestamp of the last processed row
                last_processed_ts = new_rows_df['ts'].max()

        except FileNotFoundError:
             logger.error(f"File not found during processing. It may have been rotated.")
        except Exception as e:
            logger.error(f"An error occurred in the prediction loop: {e}", exc_info=True)

        time.sleep(LOOP_INTERVAL_SECONDS)

if __name__ == "__main__":
    run_predictor() 