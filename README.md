Big Mart Sales – End-to-End Pipeline

Wrangling → Clean Training/Test → Model Selection → Hyperparameter Tuning → Final Training → Predictions

This project builds a reproducible tabular ML pipeline for the Big Mart Sales problem using pure scikit-learn. It includes:

A data wrangler that cleans and imputes raw CSVs and produces fully numeric, model-ready datasets.

A training pipeline that normalizes features, runs feature-selection diagnostics, performs KFold GridSearch on a 35% validation subset, retrains on all training data with the best hyperparameters, and exports predictions/artifacts.

1) Repository Layout & Paths

The code uses relative paths assuming the scripts live one level below the data/ and output/ folders.
If your structure differs, adjust the paths or run from the folder that satisfies the ../data and ../output locations.

repo-root/
├─ data/
│  ├─ train.csv
│  └─ test.csv
├─ output/                      # created by the pipeline; final artifacts live here
│  ├─ clean_train.csv
│  ├─ clean_test.csv
│  ├─ scaler_<model>.joblib
│  ├─ selected_features_<model>.json
│  ├─ best_params_<model>.json
│  ├─ model_<model>.joblib
│  ├─ train_predictions_<model>.csv
│  └─ test_predictions_<model>.csv
└─ code/ (or scripts/)
   ├─ data_wrangling_script.py  # contains the Wrangler class (your “Data wrangling” code)
   └─ training_script.py        # contains the BigMartTrainer class (your “TRAINING” code)


The wrangler reads from ../data/train.csv and ../data/test.csv and writes clean CSVs to ../output/.

The trainer reads ../output/clean_train.csv and ../output/clean_test.csv and writes all artifacts back to ../output/.

2) Environment & Dependencies

Python: 3.9+ recommended

Packages:

pandas, numpy, scipy

scikit-learn (>= 1.1 recommended)

joblib

Install:

pip install -U pandas numpy scipy scikit-learn joblib

3) What the Pipeline Does (Overview)
Data Wrangling (Wrangler class)

Load train/test separately (avoid leakage).

Item_Visibility: treat zeros as missing (→ NaN), cast numeric.

Fill “mappable” NaNs deterministically when a group uniquely determines a value (e.g., Outlet_Size by Outlet_Identifier; Item_Weight by Item_Identifier). Otherwise, use group mode.

Deterministic Case-A imputation: For a target and multiple predictors, build one-to-one maps where cross-tabs show uniqueness; fill missing in a non-overwriting sequence.

Item_Visibility (hierarchical): median by (Item_Identifier, Outlet_Identifier) pairs with sufficient support → fallback median by Outlet_Identifier (robust, low-variance).

Label normalization: Item_Fat_Content spelling fixes; NC* items set to Inedible.

Precision-aware casting: convert integer columns to float32 only if casting float64→float32 is safe across current floats; otherwise keep float64.

String encoding: per-column sorted unique → code 0..n-1 (stored as float64) to obtain a fully numeric matrix.

Save clean_train.csv and clean_test.csv in ../output/.

Known trade-off: independent string encodings in train and test can yield different numeric codes. Tree models (RF/HGB) are resilient; linear/SVR may interpret codes as ordinal. If needed, persist the train mapping and reuse in test.

Model Training (BigMartTrainer class)

Load clean_train.csv / clean_test.csv.

Normalize features (fit StandardScaler on all train, reuse for test). Saved to disk.

Feature-selection diagnostics (on normalized train):

|Pearson correlation|, Mutual Information, F-test, RF importance.

Final setup uses all features (robust with tree models; avoids dropping signal under numeric encodings). Selected list saved to disk.

Hyperparameter tuning (GridSearchCV + KFold) on only 35% of the training data (as required). Scoring: neg RMSE.

Final training with the best params on 100% of training data.

Report metrics: Train R² / RMSE / MAE + 5-fold CV R² & RMSE on the full training set.

Export artifacts and train/test predictions (3 columns: Item_Identifier, Outlet_Identifier, Item_Outlet_Sales).

4) Running the Pipeline

Make sure you run from the folder where the ../data and ../output relative paths are valid (e.g., from within code/).

4.1 Data Wrangling
python data_wrangling_script.py


This executes:

Wrangler.process_data(type_="train") → writes ../output/clean_train.csv

Wrangler.process_data(type_="test") → writes ../output/clean_test.csv

Console prints include NaN counts and imputation stats.

4.2 Training & Prediction
python training_script.py


At the bottom of the script you’ll see:

# Models: "rf", "hgb", "linreg", "bayes", "svr", "ridge"
m = BigMartTrainer(modelName="hgb", valSize=0.35)
m.feature_selection()
m.hyperparameter_tuning()
m.training()
m.dump_train_predictions()
m.dump_test_predictions()


To choose a different model or validation size, edit the arguments:

m = BigMartTrainer(modelName="rf", valSize=0.35)   # RandomForest
# or:
m = BigMartTrainer(modelName="ridge", valSize=0.35)

5) Outputs & Artifacts (in ../output/)

Clean data:

clean_train.csv, clean_test.csv

Training artifacts (per selected modelName):

scaler_<model>.joblib – StandardScaler fit on train (reused for test)

selected_features_<model>.json – final feature list used (currently “all”)

best_params_<model>.json – GridSearchCV best hyperparameters

model_<model>.joblib – trained model on 100% of train

Predictions:

train_predictions_<model>.csv – with Item_Identifier, Outlet_Identifier, Item_Outlet_Sales

test_predictions_<model>.csv – same schema (submission-ready)

6) What Gets Printed

The training script prints (and only this set):

Selected model name

Feature selection results & final chosen features (if uncommented)

Hyperparameter tuning best params

Training metrics: Train R² / RMSE / MAE and 5-fold CV R² / RMSE

In the provided code, some feature-selection prints are commented to keep output minimal. Uncomment if you want to see the top-k lists from each selector.

7) Why These Models Win Here (Quick Rationale)

HistGradientBoostingRegressor (HGB):
Captures nonlinearities & interactions common in retail demand; built-in regularization; fast and strong on tabular data.

RandomForestRegressor (RF):
A robust baseline that averages many decorrelated trees; handles mixed feature effects, outliers, and noisy signals gracefully.

Why not plain Linear/Ridge/BayesianRidge/SVR as final choices?

Linear models underfit nonlinear price/outlet effects without extensive feature crosses.

SVR can be powerful but is sensitive to kernel/scale and less scalable; it under-performed against HGB/RF in RMSE here.

8) Reproducibility & Anti-Leakage

Random seeds fixed at random_state=42 where applicable.

Train/test wrangling separated to avoid leakage of train distributions into test.

Scaler is fit on train only, reused on test.

GridSearch uses only the 35% validation split, as required, then final training is on all training rows.

9) Troubleshooting

Path errors: Ensure your working directory makes ../data and ../output resolve correctly.

Missing ID columns: The trainer asserts Item_Identifier and Outlet_Identifier exist in clean train/test.

Different category codes between train/test:

Tree models are resilient.

For strict alignment (esp. linear/SVR), persist train encodings and apply to test (mapping JSON).

Performance/memory: Shrink RF/HGB grids if your machine is resource-limited; reduce n_splits or n_estimators.

10) Extending the Project

Add train-mapping-based encoders (persist string→code maps from train; apply to test).

Try target log-transform + RMSLE scoring if you prefer a multiplicative error metric.

Add CatBoost or LightGBM for potentially higher leaderboard scores (native categorical support / faster training).

Introduce probabilistic imputation for mixed groups based on observed distributions when deterministic rules don’t apply.

11) Minimal Example (Interactive)
from training_script import BigMartTrainer

m = BigMartTrainer(modelName="rf", valSize=0.35)
m.feature_selection()         # normalize + diagnostics (saves scaler & selected features)
m.hyperparameter_tuning()     # GridSearchCV on 35% validation subset with KFold
m.training()                  # fit best model on 100% train; prints scores
m.dump_train_predictions()    # write train predictions (3 cols)
m.dump_test_predictions()     # write test predictions (3 cols)
