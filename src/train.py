import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge, Ridge
from sklearn.svm import SVR

class BigMartTrainer:
	def __init__(self, modelName: str = "ridge", valSize: float = 0.35):
		# Paths
		self.outputDir = os.path.join("..", "output")
		self.trainPath = os.path.join(self.outputDir, "clean_train.csv")
		self.testPath  = os.path.join(self.outputDir, "clean_test.csv")

		# Read data
		self.trainDF = pd.read_csv(self.trainPath)
		self.testDF  = pd.read_csv(self.testPath)

		# Identify target and features
		self.targetCol = self.trainDF.columns[-1]
		self.featureCols = [c for c in self.trainDF.columns[:-1]]

		# Keep IDs for outputs
		self.idCols = ["Item_Identifier", "Outlet_Identifier"]
		for c in self.idCols:
			if c not in self.trainDF.columns or c not in self.testDF.columns:
				raise ValueError(f"Required ID column '{c}' not found in train/test.")

		# Chosen model & validation size
		self.modelName = modelName.lower().strip()
		self.valSize = float(valSize)

		# Placeholders
		self.scaler = None
		self.XScaled = None
		self.testXScaled = None
		self.featuresSelected = self.featureCols[:]  # will refine after feature selection
		self.bestParams = None
		self.model = None

		print(f"Selected model: {self.modelName}")

	def feature_selection(self):
		# ----- Normalize X once on ALL training data (reused everywhere) -----
		X = self.trainDF[self.featureCols].values.astype(float)
		y = self.trainDF[self.targetCol].values.astype(float)

		self.scaler = StandardScaler()
		self.XScaled = self.scaler.fit_transform(X)
		self.testXScaled = self.scaler.transform(self.testDF[self.featureCols].values.astype(float))

		# Dump scaler for reuse
		scalerPath = os.path.join(self.outputDir, f"scaler_{self.modelName}.joblib")
		joblib.dump(self.scaler, scalerPath)

		# ----- Run feature selection tests on ALL data (normalized X) -----
		nFeatures = self.XScaled.shape[1]
		kTop = max(10, int(np.sqrt(nFeatures)))  # balanced cap to avoid overfitting

		# 1) Pearson correlation magnitude
		corrVals = []
		for j in range(nFeatures):
			xj = self.XScaled[:, j]
			# Safe guard for zero variance
			if np.std(xj) < 1e-12:
				corr = 0.0
			else:
				corr = np.corrcoef(xj, y)[0, 1]
			corrVals.append(abs(corr))
		corrOrder = np.argsort(corrVals)[::-1][:kTop]
		corrFeatures = [self.featureCols[i] for i in corrOrder]
		# print("Feature selection - Top by |Pearson corr|:", corrFeatures)

		# 2) Mutual Information (nonlinear)
		miVals = mutual_info_regression(self.XScaled, y, random_state=42)
		miOrder = np.argsort(miVals)[::-1][:kTop]
		miFeatures = [self.featureCols[i] for i in miOrder]
		# print("Feature selection - Top by Mutual Information:", miFeatures)

		# 3) F-test (linear signal vs variance)
		fVals, _ = f_regression(self.XScaled, y)
		fVals = np.nan_to_num(fVals, nan=0.0, posinf=0.0, neginf=0.0)
		fOrder = np.argsort(fVals)[::-1][:kTop]
		fFeatures = [self.featureCols[i] for i in fOrder]
		# print("Feature selection - Top by F-statistics:", fFeatures)

		# 4) Model-based importance (RandomForest, robust to monotone transforms)
		rfProbe = RandomForestRegressor(
			n_estimators=300, max_features="sqrt", min_samples_leaf=2, random_state=42, n_jobs=-1
		)
		rfProbe.fit(self.XScaled, y)
		imp = rfProbe.feature_importances_
		impOrder = np.argsort(imp)[::-1][:kTop]
		rfFeatures = [self.featureCols[i] for i in impOrder]
		# print("Feature selection - Top by RF importance:", rfFeatures)

		# Final selected features = UNION of the kTop sets (captures linear + nonlinear)
		selectedSet = set(corrFeatures) | set(miFeatures) | set(fFeatures) | set(rfFeatures)
		self.featuresSelected = [f for f in self.featureCols if f in selectedSet]
		self.featuresSelected = self.featureCols

		# Reasoning & final list
		# print("Reasoning: We take the UNION of top-k from correlation (linear), MI (nonlinear), "
		# 	  "F-test (signal-to-noise), and RF importance (interaction/nonlinearity). "
		# 	  "Using union reduces the risk of discarding real signal and helps prevent "
		# 	  "overfitting by capping k at ~sqrt(d) per method.")
		# print("Selected features (final):", self.featuresSelected)

		# Persist selected feature names (to reuse in test / future)
		with open(os.path.join(self.outputDir, f"selected_features_{self.modelName}.json"), "w", encoding="utf-8") as f:
			json.dump(self.featuresSelected, f, ensure_ascii=False, indent=2)

	def hyperparameter_tuning(self):
		# Prepare validation subset (35% of training data)
		XSel = self.trainDF[self.featuresSelected].values.astype(float)
		XSelScaled = self.scaler.transform(XSel)  # same scaler
		y = self.trainDF[self.targetCol].values.astype(float)

		XTrain, XVal, yTrain, yVal = train_test_split(
			XSelScaled, y, test_size=self.valSize, random_state=42, shuffle=True
		)

		# KFold CV on the validation subset only (per requirement)
		cv = KFold(n_splits=10, shuffle=True, random_state=42)

		# Choose estimator + param grid
		if self.modelName == "rf":
			est = RandomForestRegressor(random_state=42, n_jobs=-1)
			grid = {
				"n_estimators": [200, 400, 800],
				"max_depth": [None, 10, 20, 30],
				"min_samples_leaf": [1, 2, 4],
				"max_features": ["sqrt", 0.3, 0.5, None]
			}
		elif self.modelName == "hgb":
			est = HistGradientBoostingRegressor(random_state=42)
			grid = {
				"learning_rate": [0.03, 0.06, 0.1],
				"max_iter": [300, 600, 900],
				"max_depth": [None, 8, 12],
				"l2_regularization": [0.0, 1e-3, 1e-2],
				"max_leaf_nodes": [31, 63, 127]
			}
		elif self.modelName == "linreg":
			est = LinearRegression()
			grid = {
				"fit_intercept": [True, False],
				"positive": [False, True]
			}
		elif self.modelName == "bayes":
			est = BayesianRidge()
			grid = {
				"alpha_1": [1e-6, 1e-5, 1e-4],
				"alpha_2": [1e-6, 1e-5, 1e-4],
				"lambda_1": [1e-6, 1e-5, 1e-4],
				"lambda_2": [1e-6, 1e-5, 1e-4],
				"n_iter": [300, 600, 1000]
			}
		elif self.modelName == "svr":
			est = SVR()
			grid = {
				"kernel": ["rbf", "linear", "poly"],
				"C": [1, 3, 10, 30, 100],
				"epsilon": [0.01, 0.05, 0.1, 0.2, 0.5],
				"gamma": ["scale", "auto"]
			}
		elif self.modelName == "ridge":
			est = Ridge(random_state=42)
			grid = {
				"alpha": [0.1, 0.3, 1, 3, 10, 30, 100, 300],
				"fit_intercept": [True, False],
				"solver": ["auto", "svd", "cholesky", "lsqr", "sag", "saga"]
			}
		else:
			raise ValueError("Unknown modelName. Use one of: rf, hgb, linreg, bayes, svr, ridge")

		gs = GridSearchCV(
			estimator=est,
			param_grid=grid,
			scoring="neg_root_mean_squared_error",
			cv=cv,
			n_jobs=-1,
			refit=True
		)
		gs.fit(XVal, yVal)

		self.bestParams = gs.best_params_
		print("Hyperparameter tuning - best params:", self.bestParams)

		# Persist best params
		with open(os.path.join(self.outputDir, f"best_params_{self.modelName}.json"), "w", encoding="utf-8") as f:
			json.dump(self.bestParams, f, ensure_ascii=False, indent=2)

	def training(self):
		# Refit chosen model with best params on ALL training data
		if self.bestParams is None:
			raise RuntimeError("Run hyperparameter_tuning() before training().")

		XSel = self.trainDF[self.featuresSelected].values.astype(float)
		XSelScaled = self.scaler.transform(XSel)
		y = self.trainDF[self.targetCol].values.astype(float)

		# Build final estimator from best params
		if self.modelName == "rf":
			self.model = RandomForestRegressor(random_state=42, n_jobs=-1, **self.bestParams)
		elif self.modelName == "hgb":
			self.model = HistGradientBoostingRegressor(random_state=42, **self.bestParams)
		elif self.modelName == "linreg":
			self.model = LinearRegression(**self.bestParams)
		elif self.modelName == "bayes":
			self.model = BayesianRidge(**self.bestParams)
		elif self.modelName == "svr":
			self.model = SVR(**self.bestParams)
		elif self.modelName == "ridge":
			self.model = Ridge(random_state=42, **self.bestParams)
		else:
			raise ValueError("Unknown modelName during training.")

		self.model.fit(XSelScaled, y)

		# Report training scores + a 5-fold CV to guard against overfitting
		yPredTrain = self.model.predict(XSelScaled)
		r2 = r2_score(y, yPredTrain)
		rmse = np.sqrt(mean_squared_error(y, yPredTrain))
		mae = mean_absolute_error(y, yPredTrain)

		cv = KFold(n_splits=5, shuffle=True, random_state=42)
		cv_rmse = -cross_val_score(self.model, XSelScaled, y, scoring="neg_root_mean_squared_error", cv=cv, n_jobs=-1)
		cv_r2   = cross_val_score(self.model, XSelScaled, y, scoring="r2", cv=cv, n_jobs=-1)

		print("Training results:")
		print(f"  Train R2:  {r2:.5f}")
		print(f"  Train RMSE:{rmse:.5f}")
		print(f"  Train MAE: {mae:.5f}")
		print(f"  CV R2 mean:   {cv_r2.mean():.5f} ± {cv_r2.std():.5f}")
		print(f"  CV RMSE mean: {cv_rmse.mean():.5f} ± {cv_rmse.std():.5f}")

		# Persist trained model
		joblib.dump(self.model, os.path.join(self.outputDir, f"model_{self.modelName}.joblib"))

	def dump_train_predictions(self):
		# Predict on ALL train rows
		XSel = self.trainDF[self.featuresSelected].values.astype(float)
		XSelScaled = self.scaler.transform(XSel)
		yHat = self.model.predict(XSelScaled)

		outDF = pd.DataFrame({
			"Item_Identifier": self.trainDF["Item_Identifier"].values,
			"Outlet_Identifier": self.trainDF["Outlet_Identifier"].values,
			"Item_Outlet_Sales": yHat
		})
		outPath = os.path.join(self.outputDir, f"train_predictions_{self.modelName}.csv")
		outDF.to_csv(outPath, index=False)

	def dump_test_predictions(self):
		# Predict on test
		XSelTest = self.testDF[self.featuresSelected].values.astype(float)
		XSelTestScaled = self.scaler.transform(XSelTest)
		yTestHat = self.model.predict(XSelTestScaled)

		outDF = pd.DataFrame({
			"Item_Identifier": self.testDF["Item_Identifier"].values,
			"Outlet_Identifier": self.testDF["Outlet_Identifier"].values,
			"Item_Outlet_Sales": yTestHat
		})
		outPath = os.path.join(self.outputDir, f"test_predictions_{self.modelName}.csv")
		outDF.to_csv(outPath, index=False)

if __name__ == "__main__":
	# Models: "rf", "hgb", "linreg", "bayes", "svr", "ridge"
	m = BigMartTrainer(modelName = "hgb", valSize = 0.35)
	m.feature_selection()
	m.hyperparameter_tuning()
	m.training()
	m.dump_train_predictions()
	m.dump_test_predictions()