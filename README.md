<div align="center">

# 🌱 AgroNutrient AI
### Precision Fertilizer Recommendation System `v1.0`

An end-to-end Machine Learning web application designed to recommend optimal crop fertilizers using multi-feature soil chemistry, environmental telemetry, and Scikit-Learn tree inference.

[![Live Demo](https://img.shields.io/badge/Live_App-Netlify-00C7B7?style=for-the-badge&logo=netlify&logoColor=white)](https://adorable-sunflower-8d6f45.netlify.app/)
[![API Status](https://img.shields.io/badge/API-Render_Cloud-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://fertilizer-recommendation-backend.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Tailwind CSS](https://img.shields.io/badge/UI-TailwindCSS-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white)](https://tailwindcss.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](https://opensource.org/licenses/MIT)

<br />

### 🌐 [Launch Live Web Application →](https://adorable-sunflower-8d6f45.netlify.app/)
### 📑 [Interactive API Docs (Swagger UI) →](https://fertilizer-recommendation-backend.onrender.com/docs)

</div>

---

## 📑 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [Feature Matrix & Parameter Specs](#-feature-matrix--parameter-specs)
- [API Specification](#-api-specification)
  - [1. Dynamic Options Loader (`/api/options`)](#1-dynamic-options-loader-apioptions)
  - [2. Fertilizer Inference Engine (`/api/predict`)](#2-fertilizer-inference-engine-apipredict)
- [Local Setup & Installation](#-local-setup--installation)
- [Production Deployment](#-production-deployment)
  - [Backend on Render](#backend-on-render)
  - [Frontend on Netlify](#frontend-on-netlify)
- [License & Credits](#-license--credits)

---

## 🌾 Project Overview

Soil nutrient depletion and improper fertilizer usage severely degrade soil health and decrease agricultural yield. **AgroNutrient AI** bridges this gap by calculating the optimal chemical or organic fertilizer formulation required for a given target crop based on immediate environmental metrics, macro-nutrients ($N$, $P$, $K$), organic carbon, and soil moisture levels.

The platform is designed with a **decoupled, API-first architecture**:
1. **Frontend:** Lightweight, zero-dependency Tailwind CSS single-page client hosted on **Netlify**.
2. **Backend:** Asynchronous, high-throughput REST service built with **FastAPI** and hosted on **Render**.
3. **Machine Learning:** Optimized multi-class **Decision Tree Classifier** serialized alongside respective categorical encoders using Pickle.

---

## ⚡ Key Features

| Feature | Details |
| :--- | :--- |
| 🧪 **Multi-Nutrient Analysis** | Evaluates Nitrogen ($N$), Phosphorous ($P$), Potassium ($K$), and Soil Organic Carbon alongside pH balance. |
| 🌦️ **Microclimate Integration** | Ingests ambient temperature, relative moisture, and rainfall parameters for climate-aware recommendations. |
| 🔄 **Dynamic Taxonomy Sync** | Dropdowns for Soil and Crop types populate dynamically via `/api/options`, avoiding static frontend hardcoding. |
| 🛡️ **Pydantic Schema Validation** | Strict type casting, floating-point bounds checking, and categorical containment validation. |
| ⚡ **Sub-Millisecond Inference** | Serialized decision tree structures deliver rapid, lightweight classification without expensive GPU dependencies. |
| 📱 **Responsive Mobile Client** | Modern utility-first interface styled with Tailwind CSS, supporting accessible inputs and dynamic state alerts. |

---

## 🏗️ System Architecture

```text
       [ User / Agriculturalist ]
                   │
                   ▼
      ┌─────────────────────────┐
      │     Netlify Client      │
      │   Tailwind + Fetch API  │
      └────────────┬────────────┘
                   │
                   │ (HTTP GET /api/options)
                   │ (HTTP POST /api/predict)
                   ▼
      ┌─────────────────────────┐
      │   Render Cloud Service  │
      │      FastAPI + CORS     │
      └────────────┬────────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
┌──────────────────┐ ┌──────────────────┐
│  Pydantic Engine │ │ Scikit-Learn Pipeline│
│ Type & Value     │ │ - Categorical Decode│
│ Validation       │ │ - Feature Alignment│
│                  │ │ - Tree Inference   │
└──────────────────┘ └──────────────────┘
```

---

## 📁 Repository Structure

```plaintext
fertilizer_project/
├── backend/
│   ├── main.py                                # FastAPI application entry point & routes
│   ├── train_model.py                         # Model training and artifact generation pipeline
│   ├── requirements.txt                       # Backend Python dependencies
│   ├── model.pkl                              # Serialized Decision Tree classifier
│   ├── soil_encoder.pkl                       # LabelEncoder for soil classes
│   ├── crop_encoder.pkl                       # LabelEncoder for crop varieties
│   ├── fertilizer_encoder.pkl                 # LabelEncoder for output fertilizers
│   └── __pycache__/                           # Cached compiled bytecode
├── frontend/
│   └── index.html                             # Responsive Tailwind UI client
├── data/
│   └── fertilizer_recommendation_dataset.csv  # Cleaned agro-chemical training dataset
├── fartilizer_recomendatin/                   # Python virtual environment (ignored in git)
├── .vscode/                                   # Workspace preferences
├── .git/                                      # Git version control directory
└── .gitignore                                 # Git ignore filters
```

---

## 📊 Feature Matrix & Parameter Specs

The predictive model processes 10 distinct features to produce the optimal fertilizer recommendation:

| Feature | Data Type | Units / Format | Typical Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| Temperature | Continuous (float) | °C | 10.0 – 45.0 | Ambient environmental temperature |
| Moisture | Continuous (float) | Fraction / Ratio | 0.0 – 1.0 | Current soil moisture saturation |
| Rainfall | Continuous (float) | mm | 50.0 – 350.0 | Local precipitation level |
| PH | Continuous (float) | pH Index (0–14) | 4.0 – 9.0 | Soil acidity/alkalinity scale |
| Nitrogen (N) | Continuous (float) | Ratio / ppm | 0.0 – 150.0 | Readily available soil Nitrogen |
| Phosphorous (P) | Continuous (float) | Ratio / ppm | 0.0 – 150.0 | Readily available soil Phosphorus |
| Potassium (K) | Continuous (float) | Ratio / ppm | 0.0 – 150.0 | Readily available soil Potassium |
| Carbon | Continuous (float) | Percentage (%) | 0.1 – 5.0 | Organic carbon content percentage |
| Soil | Categorical (str) | Class label | String | Soil type (e.g., Loamy Soil, Clayey) |
| Crop | Categorical (str) | Class label | String | Target crop intended for cultivation |

---

## 🔌 API Specification

### 1. Dynamic Options Loader (`/api/options`)

Queries trained label encoders to return valid soil categories and supported crop varieties.

**URL:** `/api/options`  
**Method:** `GET`  
**Content-Type:** `application/json`

**Response (200 OK)**

```json
{
  "soils": [
    "Black",
    "Clayey",
    "Loamy",
    "Red",
    "Sandy"
  ],
  "crops": [
    "Barley",
    "Cotton",
    "Ground Nuts",
    "Maize",
    "Millets",
    "Oil seeds",
    "Paddy",
    "Pulses",
    "Sugarcane",
    "Tobacco",
    "Wheat"
  ]
}
```

### 2. Fertilizer Inference Engine (`/api/predict`)

Executes tabular inference through the Decision Tree classifier to output the target fertilizer formulation.

**URL:** `/api/predict`  
**Method:** `POST`  
**Content-Type:** `application/json`

**Request Payload**

```json
{
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
```

**Successful Response (200 OK)**

```json
{
  "status": "success",
  "recommended_fertilizer": "Urea"
}
```

**Validation Error (400 Bad Request)**

```json
{
  "detail": "Invalid soil type: Peat Soil"
}
```

---

## 💻 Local Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/fertilizer_project.git
cd fertilizer_project
```

### 2. Virtual Environment Configuration

```bash
# Windows
python -m venv fartilizer_recomendatin
fartilizer_recomendatin\Scripts\activate

# Linux / macOS
python3 -m venv fartilizer_recomendatin
source fartilizer_recomendatin/bin/activate
```

### 3. Install Dependencies

```bash
cd backend
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Train the Model Pipeline

Run the script to train the model on the local dataset and produce the .pkl artifact binaries:

```bash
python train_model.py
```

### 5. Launch FastAPI Backend

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open http://127.0.0.1:8000/docs to test via Swagger UI.

### 6. Serve the Frontend Locally

In a separate terminal, serve the frontend:

```bash
cd ../frontend
python -m http.server 3000
```

Visit http://localhost:3000 in your web browser. (Note: Update `API_BASE` in `index.html` to `http://127.0.0.1:8000` for offline local development).

---

## ☁️ Production Deployment

### Backend on Render

1. Push your code to GitHub.
2. In the Render Dashboard, select **New +** → **Web Service**.
3. Link your repository and set the following parameters:
   - **Root Directory:** `backend`
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Deploy the service and note your live API base URL.

### Frontend on Netlify

1. Log in to Netlify and select **Add new site** → **Deploy manually** (or connect your GitHub repository).
2. Choose the `frontend/` directory as the publish root.
3. Ensure the `API_BASE` constant in `frontend/index.html` references your live Render backend URL:

```javascript
const API_BASE = "https://fertilizer-recommendation-backend.onrender.com";
```

4. Deploy site.

---

## 📄 License & Credits

- **Machine Learning Engine:** Scikit-Learn & Pandas
- **Web Layer:** FastAPI & Starlette
- **Styling:** Tailwind CSS

Distributed under the **MIT License**.
