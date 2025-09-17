import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.calibration import CalibratedClassifierCV
import joblib
import xgboost as xgb

data = pd.read_csv("Crop_recommendation_with_season.csv")

data["NPK_ratio"] = data["N"] / (data["P"] + data["K"] + 1e-5)

def get_rainfall_bucket(rainfall):
    if rainfall <= 50:
        return "low"
    elif rainfall <= 200:
        return "medium"
    else:
        return "high"

data["rainfall_bucket"] = data["rainfall"].apply(get_rainfall_bucket)

# Prepare Features & Target
num_features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "NPK_ratio"]

season_dummies = pd.get_dummies(data["season"], prefix="season")
rainfall_dummies = pd.get_dummies(data["rainfall_bucket"], prefix="rainfall")

X = pd.concat([data[num_features], season_dummies, rainfall_dummies], axis=1)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data["label"])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost Model
xgb_model = xgb.XGBClassifier(
    n_estimators=1200,
    learning_rate=0.03,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2,             # L2 regularization
    reg_alpha=1,              # L1 regularization
    objective="multi:softprob",
    num_class=len(label_encoder.classes_),
    eval_metric="mlogloss",
    random_state=42,
    tree_method="hist",
    use_label_encoder=False
)

# Wrap with CalibratedClassifierCV to improve probability calibration
model = CalibratedClassifierCV(xgb_model, method="isotonic", cv=3)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred_proba = model.predict_proba(X_test_scaled)
y_pred = np.argmax(y_pred_proba, axis=1)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Log Loss:", log_loss(y_test, y_pred_proba))
print("\nClassification Report:\n",
      classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save Model & Objects
joblib.dump(model, "crop_recommendation_model_xgb.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

print("\nCalibrated XGBoost, Scaler, Label Encoder & Features Saved Successfully!")
