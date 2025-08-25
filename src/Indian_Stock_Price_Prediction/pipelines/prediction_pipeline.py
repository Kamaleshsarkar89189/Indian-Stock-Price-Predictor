# prediction_pipeline.py

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from src.Indian_Stock_Price_Prediction.components.data_ingestion import fetch_stock_data
from src.Indian_Stock_Price_Prediction.components.data_transformation import transform_data
from src.Indian_Stock_Price_Prediction.pipelines.training_pipeline import train_pipeline

def predict_pipeline_using_training(
	ticker: str,
	prediction_date: str,
	lookback_days: int = 365,
	test_size: float = 0.2,
	prediction_days: int = 1,
	scale: bool = True,
	ensemble: bool = True
) -> Dict[str, Any]:
	"""
	Build predictions by leveraging train_pipeline for training/evaluation,
	then recomputing the latest feature row for inference.
	"""
	# Train and get trained models + performance
	train_out = train_pipeline(
		ticker=ticker,
		prediction_date=prediction_date,
		lookback_days=lookback_days,
		test_size=test_size,
		prediction_days=prediction_days,
		scale=scale
	)

	trained_models = train_out["trained_models"]
	performance_df = train_out["performance"]
	feature_names = train_out["feature_names"]

	# Recompute features to get the latest test row for inference
	df = fetch_stock_data(ticker=ticker, prediction_date=prediction_date, lookback_days=lookback_days)
	price_col = "adj_close" if "adj_close" in df.columns else "close"
	current_price: Optional[float] = float(df[price_col].iloc[-1]) if price_col in df.columns else None

	if not 0 < test_size < 1:
		raise ValueError("test_size must be between 0 and 1")
	split_idx = int(len(df) * (1 - test_size))
	train_df = df.iloc[:split_idx].reset_index(drop=True)
	test_df = df.iloc[split_idx:].reset_index(drop=True)

	X_train, X_test, y_train, y_test, scaler, feat_names_2 = transform_data(
		train_df=train_df,
		test_df=test_df,
		target_col=None,
		prediction_days=prediction_days,
		scale=scale
	)
	# Align with training feature order if needed
	if feature_names and list(feature_names) != list(feat_names_2):
		# Attempt to align columns by name; if not available, use current order
		try:
			idx = [feat_names_2.index(f) for f in feature_names]
			X_test_aligned = X_test[:, idx]
		except Exception:
			X_test_aligned = X_test
	else:
		X_test_aligned = X_test

	# Predict next target using last available feature row
	last_feature_row = X_test_aligned[-1].reshape(1, -1)
	predictions: Dict[str, float] = {}
	for name, model in trained_models.items():
		try:
			predictions[name] = float(model.predict(last_feature_row)[0])
		except Exception:
			predictions[name] = np.nan

	ensemble_pred: Optional[float] = None
	if ensemble:
		valid = [v for v in predictions.values() if v is not None and np.isfinite(v)]
		ensemble_pred = float(np.mean(valid)) if valid else None

	return {
		"predictions": predictions,
		"ensemble_prediction": ensemble_pred,
		"current_price": current_price,
		"performance": performance_df,
		"feature_names": feature_names,
		"metadata": {
			"rows": int(len(df)),
			"train_rows": int(len(train_df)),
			"test_rows": int(len(test_df)),
			"prediction_days": int(prediction_days)
		}
	}

if __name__ == "__main__":
	# Example:
	# out = predict_pipeline_using_training("RELIANCE.NS", "2025-08-01", 365, 0.2, prediction_days=1)
	# print("Ensemble:", out["ensemble_prediction"])
	# print(pd.DataFrame.from_dict(out["predictions"], orient="index", columns=["next_target"]))
	pass