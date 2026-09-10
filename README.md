Markdown# 🌱 Fertilizer Recommendation System

An end-to-end Machine Learning web application designed to recommend optimal fertilizers based on soil chemistry, crop requirements, and environmental parameters. Built with a decoupled architecture featuring a **Tailwind CSS** client hosted on Netlify, a **FastAPI** inference service hosted on Render, and a **Scikit-Learn** classifier.

---

## 🔗 Live Deployments

- **Live Web Application:** [https://adorable-sunflower-8d6f45.netlify.app/](https://adorable-sunflower-8d6f45.netlify.app/)
- **API Documentation (Swagger UI):** [https://fertilizer-recommendation-backend.onrender.com/docs](https://fertilizer-recommendation-backend.onrender.com/docs)
- **API Base URL:** `https://fertilizer-recommendation-backend.onrender.com`

---

## 📌 Features

- **Decoupled Architecture:** Standalone static client hosted on Netlify communicating asynchronously with a FastAPI REST API running on Render.
- **Dynamic Dropdowns:** Soil and crop select fields populate dynamically via the `/api/options` endpoint based on trained encoder classes.
- **Strict Data Validation:** Comprehensive schema validation powered by Pydantic for input chemistry and environmental parameters.
- **Fast Decision Tree Inference:** Multi-class classification mapping input soil parameters directly to targeted fertilizer recommendations.

---

## 🛠️ Tech Stack

- **Frontend:** HTML5, Tailwind CSS (CDN), Vanilla JavaScript (Fetch API) — Hosted on **Netlify**
- **Backend Framework:** FastAPI, Uvicorn — Hosted on **Render**
- **Machine Learning:** Scikit-Learn (DecisionTreeClassifier, LabelEncoder), Pandas, NumPy
- **Serialization:** Pickle
- **Environment:** Python 3.9+

---

## 📂 Project Structure

```text
fertilizer_project/
├── backend/
│   ├── main.py                     # FastAPI backend application entry point
│   ├── train_model.py              # Model training and artifact generation script
│   ├── requirements.txt            # Python dependencies
│   ├── model.pkl                   # Serialized Decision Tree model
│   ├── soil_encoder.pkl            # LabelEncoder for soil classes
│   ├── crop_encoder.pkl            # LabelEncoder for crop classes
│   ├── fertilizer_encoder.pkl      # LabelEncoder for target fertilizers
│   └── __pycache__/
├── frontend/
│   └── index.html                  # Tailwind UI client (deployed to Netlify)
├── data/
│   └── fertilizer_recommendation_dataset.csv  # Dataset used for training
├── fartilizer_recomendatin/        # Python virtual environment (ignored in git)
│   ├── pyvenv.cfg
│   ├── Include/
│   ├── Lib/
│   ├── Scripts/
│   └── ...
├── .vscode/                        # IDE configurations
├── .git/                           # Git version control directory
└── .gitignore                      # Git ignore file
⚙️ Local Setup & Installation1. PrerequisitesPython 3.9+Git2. Clone the RepositoryBashgit clone [https://github.com/your-username/fertilizer_project.git](https://github.com/your-username/fertilizer_project.git)
cd fertilizer_project
3. Backend SetupBash# Navigate to the backend directory
cd backend

# Create and activate a virtual environment
python -m venv fartilizer_recomendatin

# On Windows:
fartilizer_recomendatin\Scripts\activate

# On macOS/Linux:
source fartilizer_recomendatin/bin/activate

# Install dependencies
pip install -r requirements.txt
Train the Model (Generate Artifacts)If .pkl artifacts need to be rebuilt from data/fertilizer_recommendation_dataset.csv:Bashpython train_model.py
Run the API ServerBashuvicorn main:app --reload --host 0.0.0.0 --port 8000
The API will now be running locally at http://127.0.0.1:8000.4. Frontend SetupOpen frontend/index.html.To test against your local backend, update the API_BASE variable:JavaScriptconst API_BASE = "[http://127.0.0.1:8000](http://127.0.0.1:8000)";
(For production, ensure it points to: https://fertilizer-recommendation-backend.onrender.com)Open frontend/index.html directly in your browser or run a lightweight static server:Bashcd ../frontend
python -m http.server 3000
Access the client at http://localhost:3000.📡 API Reference1. Get Dropdown OptionsFetches all available soil and crop categories from the trained encoders.Endpoint: GET /api/optionsResponse (200 OK):JSON{
  "soils": ["Black", "Clayey", "Loamy", "Red", "Sandy"],
  "crops": ["Barley", "Cotton", "Ground Nuts", "Maize", "Millets", "Oil seeds", "Paddy", "Pulses", "Sugarcane", "Tobacco", "Wheat"]
}
2. Predict Recommended FertilizerAccepts soil chemistry and weather parameters to return the best-matching fertilizer.Endpoint: POST /api/predictRequest Body:JSON{
  "temperature": 25.0,
  "moisture": 0.5,
  "rainfall": 200.0,
  "ph": 6.5,
  "nitrogen": 50.0,
  "phosphorous": 50.0,
  "potassium": 50.0,
  "carbon": 1.0,
  "soil": "Loamy Soil",
  "crop": "rice"
}
Response (200 OK):JSON{
  "status": "success",
  "recommended_fertilizer": "Urea"
}
Error Response (400 Bad Request):JSON{
  "detail": "Invalid soil type: Volcanic"
}
📊 Features Used for PredictionFeatureTypeDescriptionTemperatureContinuous (float)Ambient temperature in °CMoistureContinuous (float)Soil moisture levelRainfallContinuous (float)Precipitation in mmPHContinuous (float)Soil pH scale (0–14)NitrogenContinuous (float)Available Nitrogen (N) contentPhosphorousContinuous (float)Available Phosphorous (P) contentPotassiumContinuous (float)Available Potassium (K) contentCarbonContinuous (float)Organic carbon ratioSoilCategorical (str)Soil class (e.g., Loamy Soil, Sandy)CropCategorical (str)Target crop for cultivation📄 LicenseDistributed under the MIT License. See LICENSE for details.
