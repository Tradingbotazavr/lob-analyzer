import pandas as pd
import joblib
import os
from loguru import logger

MODEL_PATH = "models/random_forest_model.pkl"

def load_model(model_path: str):
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return None, None

    artifacts = joblib.load(model_path)
    model = artifacts["model"]
    features = artifacts["feature_names"]

    logger.success("Model and feature list loaded.")
    return model, features


def predict_from_dataframe(df: pd.DataFrame, model, feature_names: list[str]) -> pd.Series:
    if not all(col in df.columns for col in feature_names):
        missing = list(set(feature_names) - set(df.columns))
        logger.error(f"Missing features in input data: {missing}")
        return pd.Series(dtype=int)

    X = df[feature_names].dropna()
    if X.empty:
        logger.warning("Input has no valid rows after dropna.")
        return pd.Series(dtype=int)

    predictions = model.predict(X)
    return pd.Series(predictions, index=X.index, name="prediction")


if __name__ == "__main__":
    # Пример использования
    sample_path = "data/processed/sample_input.parquet"
    if not os.path.exists(sample_path):
        logger.error(f"Sample input not found: {sample_path}")
        exit(1)

    logger.info(f"Loading sample data from {sample_path}")
    df = pd.read_parquet(sample_path)

    model, features = load_model(MODEL_PATH)
    if model is None:
        exit(1)

    preds = predict_from_dataframe(df, model, features)
    logger.info(f"Predictions:\n{preds.value_counts().to_dict()}") 