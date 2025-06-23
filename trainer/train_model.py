import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import glob
import os
from loguru import logger

# --- Configuration ---
DATA_PATH = "data/merged/btcusdt_realtime_1750620563.parquet"
MODEL_OUTPUT_PATH = "models/random_forest_model.pkl"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Define features to be used for training
# Ensure these features are available in your dataset
FEATURES = [
    'price', 'qty', 'mid_price', 'avg_price', 'trade_count',
    'buy_volume', 'sell_volume', 'buy_sell_ratio', 'bid_volume',
    'ask_volume', 'imbalance',
    # New features
    'price_std', 'ma_volume', 'buy_vol_speed', 'sell_vol_speed',
    # Other potentially useful features (ensure they are calculated)
    'volatility', 'activity_spike', 'bid_volume_near', 'ask_volume_near',
    'bid_volume_pct_025', 'ask_volume_pct_025', 'imbalance_pct_025',
    'bid_density_lvl1_3', 'ask_density_lvl1_3', 'is_spoofing_like',
    'has_fake_wall', 'orderbook_entropy', 'orderbook_roc', 'price_roc'
]

TARGET_COLUMN = 'target'


def train_model():
    """
    Trains a RandomForestClassifier model on the latest aggregated data.
    """
    logger.info("Starting model training process...")

    # --- Load Data ---
    if not os.path.exists(DATA_PATH):
        logger.error(f"Data file not found at {DATA_PATH}. Aborting training.")
        return

    logger.info(f"Loading data from {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    logger.info(f"Loaded {len(df)} records.")

    # --- Preprocessing ---
    logger.info("Preprocessing data...")

    # Drop rows with missing target
    df.dropna(subset=[TARGET_COLUMN], inplace=True)
    logger.info(f"Removed rows with missing target. {len(df)} records remaining.")

    if df.empty:
        logger.error("No data remaining after dropping NaNs in target. Aborting.")
        return
        
    # Ensure all feature columns exist, fill missing with 0
    for col in FEATURES:
        if col not in df.columns:
            logger.warning(f"Feature column '{col}' not found. Adding it with default value 0.")
            df[col] = 0
    
    # Fill any other NaNs in feature columns
    df[FEATURES] = df[FEATURES].fillna(0)

    # Convert target to integer type if it's not
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    X = df[FEATURES]
    y = df[TARGET_COLUMN]
    
    # Verify we have data to train on
    if X.empty or y.empty:
        logger.error("No data available for training after preprocessing. Aborting.")
        return

    logger.info(f"Training with {len(X.columns)} features on {len(X)} samples.")
    logger.debug(f"Features used: {X.columns.tolist()}")

    # --- Train/Test Split ---
    logger.info(f"Splitting data into training and testing sets (test_size={TEST_SIZE}).")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y if len(y.unique()) > 1 else None
    )
    
    if len(y_train.unique()) < 2:
        logger.error(f"Training target has only {len(y_train.unique())} unique values. Cannot train classifier. Aborting.")
        return
    
    if len(y_test) == 0:
        logger.error("Test set is empty. Cannot perform evaluation. Consider reducing TEST_SIZE or collecting more data.")
        # We can still train the model and save it, just skip evaluation
    else:
        # --- Model Training ---
        logger.info("Training RandomForestClassifier...")
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            class_weight='balanced',
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        logger.info("Model training completed.")

        # --- Evaluation ---
        logger.info("Evaluating model performance...")
        y_pred = model.predict(X_test)

        logger.info("\n" + classification_report(y_test, y_pred))
        logger.info("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)))

        # Feature Importance
        feature_importance = pd.DataFrame({
            'feature': FEATURES,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        logger.info("Feature Importances:\n" + str(feature_importance))


        # --- Save Model ---
        os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
        joblib.dump(model, MODEL_OUTPUT_PATH)
        logger.info(f"Model saved to {MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    train_model()
