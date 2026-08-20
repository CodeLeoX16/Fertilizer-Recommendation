import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Resolve dataset path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "..", "data", "fertilizer_recommendation_dataset.csv")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}")

df = pd.read_csv(data_path)
df.columns = df.columns.str.strip()

# 2. Encode categorical variables
soil_encoder = LabelEncoder()
crop_encoder = LabelEncoder()
fert_encoder = LabelEncoder()

df["Soil"] = soil_encoder.fit_transform(df["Soil"])
df["Crop"] = crop_encoder.fit_transform(df["Crop"])
df["Fertilizer"] = fert_encoder.fit_transform(df["Fertilizer"])

# 3. Train/Test Split
feature_cols = [
    "Temperature", "Moisture", "Rainfall", "PH",
    "Nitrogen", "Phosphorous", "Potassium", "Carbon",
    "Soil", "Crop"
]
X = df[feature_cols]
y = df["Fertilizer"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Print metrics
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=fert_encoder.classes_))

# 5. Save artifacts in the backend folder
artifacts = {
    "model.pkl": model,
    "soil_encoder.pkl": soil_encoder,
    "crop_encoder.pkl": crop_encoder,
    "fertilizer_encoder.pkl": fert_encoder
}

for name, obj in artifacts.items():
    dest_path = os.path.join(BASE_DIR, name)
    with open(dest_path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved: {name}")