import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

try:
    fertilizer_data = pd.read_csv("fertilizer_extended.csv", encoding="utf-8")
except UnicodeDecodeError:
    fertilizer_data = pd.read_csv("fertilizer_extended.csv", encoding="latin1")

print("Fertilizer dataset loaded. Shape:", fertilizer_data.shape)

fertilizer_data.columns = (
    fertilizer_data.columns.str.strip()
    .str.replace("(n)", "N", regex=False)
    .str.replace("(p)", "P", regex=False)
    .str.replace("(k)", "K", regex=False)
)

# Categorize Fertilizer
def categorize_fertilizer(name):
    name = str(name).lower()
    if any(w in name for w in ["blood","bone","fish","guano","manure","feather","hoof","horn","leather","meat","scrap","milk","egg","crab","shrimp","lobster"]):
        return "Animal-based"
    if any(w in name for w in ["alfalfa","cotton","hay","straw","grain","seed","stalk","bean","pea","clover","soybean","oats","wheat","barley","corn","tobacco","tomato","fruit","leaf","leaves","grass","seaweed","kelp","compost","frass"]):
        return "Plant-based"
    if any(w in name for w in ["ash","slag","phosphate","lime","rock","greensand","coal","marl","char","soot","dust"]):
        return "Ash/Mineral-based"
    return "Other"

fertilizer_data["Category"] = fertilizer_data["Fertilizer"].apply(categorize_fertilizer)

# Features & Target
features = ["Nitrogen N", "Phosphorus P", "Potassium K"]
for col in ["temperature","humidity","ph","rainfall","season","crop_type"]:
    if col not in fertilizer_data.columns:
        fertilizer_data[col] = 0
    features.append(col)

X = fertilizer_data[features].copy()

# Encode categorical features
for col in ["season","crop_type"]:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

y = fertilizer_data["Category"]
y_enc = LabelEncoder()
y_encoded = y_enc.fit_transform(y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Weighted RandomForest
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc*100:.2f}%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=y_enc.classes_, zero_division=0))

# Save Model, Encoder & Scaler
joblib.dump(model, "fertilizer_model.pkl")
joblib.dump(y_enc, "fertilizer_encoder.pkl")
joblib.dump(scaler, "fertilizer_scaler.pkl")
joblib.dump(features, "fertilizer_features.pkl")

print("\nFertilizer Model, Encoder & Scaler Saved Successfully!")
