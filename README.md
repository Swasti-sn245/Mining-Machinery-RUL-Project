# MineGuard — Mining Equipment RUL Prediction Dashboard

An AI-powered web dashboard that predicts the **Remaining Useful Life (RUL)** of open-pit mining transportation equipment (Dumpers, Dozers, Shovels, Draglines, Graders, Scrapers), along with the top failure drivers for each prediction.

## Live Demo
🔗 **[View Live Dashboard](#)** ← *(update after deployment — see Deployment section)*

---

## What It Does

- Takes 14 engine sensor readings as input (temperatures, pressures, shaft speeds, flow/bleed ratios)
- Predicts Remaining Useful Life, scaled to real-world hours for the selected machine type
- Shows the **top 3 SHAP-driven reasons** the machine is likely to fail — genuinely unique per prediction, not a fixed global ranking
- Displays a real-time health gauge and risk level (Low / Medium / High / Critical)
- Logs prediction history

## Why NASA C-MAPSS Instead of a "Mining" Dataset

Real mining-equipment run-to-failure data is proprietary — OEMs don't release it publicly. Rather than inventing a synthetic RUL formula (a common but scientifically shaky shortcut), this project uses **NASA's C-MAPSS FD001** turbofan degradation dataset, where every engine is run in simulation **until it actually fails**. RUL here is a genuinely observed value — `(cycle at failure) − (current cycle)` — not a formula fit to make numbers look right. Sensor readings are used as illustrative analogs for heavy-machinery degradation. Full reasoning and literature citations are in `model/train_model.ipynb`.

## Model Performance (held-out test set, 100 unseen engines)

| Model | Test R² | Test MAE (cycles) | Overfit Gap |
|---|---|---|---|
| **Random Forest** (deployed) | **0.8295** | **11.89** | 0.0199 |
| XGBoost | 0.8224 | 11.71 | 0.0096 |
| Voting Ensemble | 0.8189 | 11.81 | 0.0092 |
| ANN (MLP) | 0.8177 | 11.92 | 0.0000 |
| Gradient Boosting | 0.8080 | 12.33 | 0.0192 |
| AdaBoost | 0.7865 | 13.95 | −0.0106 |

Tuned via `RandomizedSearchCV` + `GroupKFold` (no engine's cycles ever span train/val/test), cross-validated with 5-fold `GroupKFold`, selected on test performance. Full methodology in `model/train_model.ipynb`.

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | Random Forest (scikit-learn), tuned via RandomizedSearchCV + GroupKFold |
| Training Data | NASA C-MAPSS FD001 (real turbofan run-to-failure data) |
| Explainability | SHAP TreeExplainer (per-sample) |
| Backend API | FastAPI + Uvicorn/Gunicorn |
| Rate limiting | slowapi |
| Frontend | HTML + CSS + Vanilla JS (no build step) |
| Containerization | Docker + Docker Compose |
| Deployment (backend) | Render / Railway / Fly.io / Docker |
| Deployment (frontend) | Vercel / GitHub Pages / Netlify |

## Project Structure

```
mineguard/
├── backend/
│   ├── app_v4.py            ← FastAPI app (predict, health, models, history)
│   ├── config.py             ← Environment-driven settings (pydantic-settings)
│   ├── requirements.txt      ← Backend runtime dependencies only
│   ├── .env.example          ← Copy to .env to override local config
│   └── predictions.log       ← Generated at runtime (gitignored)
├── frontend/
│   ├── home.html
│   ├── rul.html              ← Main RUL predictor UI
│   ├── about.html
│   ├── explosivity.html
│   ├── shared.css
│   └── config.js             ← Resolves API base URL (localhost vs production)
├── model/
│   ├── train_model.ipynb     ← Run this to (re)train the model
│   ├── rul_model.pkl         ← Trained Random Forest (generated)
│   ├── feature_scaler.pkl    ← StandardScaler fit on training data (generated)
│   ├── shap_explainer.pkl    ← SHAP TreeExplainer (generated)
│   ├── model_metadata.json   ← Metrics, feature list, hyperparameters (generated)
│   └── requirements.txt      ← Extra deps needed only to retrain (matplotlib, jupyter)
├── dataset/
│   └── CMAPSSData/
│       ├── train_FD001.txt
│       ├── test_FD001.txt
│       └── RUL_FD001.txt
├── Dockerfile
├── docker-compose.yml
├── Procfile                  ← For Render/Railway/Heroku-style platforms
├── requirements.txt           ← Points to backend/ and model/ requirements
├── .env.example
├── .gitignore
└── .dockerignore
```

---

## Running Locally

### Prerequisites
- Python 3.12 (or 3.10+)
- pip
- ~500MB free disk space (dataset + dependencies)

### 1. Clone and enter the project
```bash
git clone https://github.com/YOUR_USERNAME/mineguard.git
cd mineguard
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Activate it:
# macOS/Linux:
source venv/bin/activate
# Windows (PowerShell):
venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
# Backend only (fastest, if the model is already trained):
pip install -r backend/requirements.txt

# Full setup (backend + retrain the model):
pip install -r backend/requirements.txt -r model/requirements.txt
```

### 4. Train the model (only if `model/*.pkl` files aren't already present)
```bash
cd model
jupyter notebook train_model.ipynb
# Run all cells top to bottom (~5-8 min on a typical laptop CPU).
# This downloads the dataset automatically if dataset/CMAPSSData/*.txt is missing,
# and saves rul_model.pkl, feature_scaler.pkl, shap_explainer.pkl, model_metadata.json.
cd ..
```

### 5. Start the backend
```bash
cd backend
cp .env.example .env   # optional — defaults work fine for local dev
uvicorn app_v4:app --reload --port 5000
```
Leave this running. Visit `http://localhost:5000/docs` to confirm it's up (interactive Swagger UI).

### 6. Open the frontend
In a **new terminal** (keep the backend running):
```bash
cd frontend
python -m http.server 8080
```
Then open `http://localhost:8080/home.html` in your browser.

> `frontend/config.js` auto-detects `localhost` and points the UI at `http://localhost:5000` — no manual edits needed for local use.

### 7. Try it
Go to the **RUL Predictor** page, click **📋 EXAMPLE** to autofill a realistic sensor reading, pick a machine type, and click **⚡ PREDICT RUL**.

---

## Running with Docker (alternative to steps 3–6 above)

Requires the model to already be trained (step 4 above) since `model/*.pkl` files are copied into the image.

```bash
docker compose up --build
```
- Backend: `http://localhost:5000`
- Frontend: `http://localhost:8080`

---

## Deployment

### Backend (Render — free tier)
1. Push this repo to GitHub (see below)
2. [render.com](https://render.com) → New → Web Service → connect your repo
3. Build Command: `pip install -r backend/requirements.txt`
4. Start Command: `cd backend && gunicorn app_v4:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT`
5. Add environment variable `ALLOWED_ORIGINS` set to your frontend's deployed URL
6. Copy the resulting Render URL

### Frontend (Vercel — free tier)
1. [vercel.com](https://vercel.com) → Import Git Repository
2. Set root directory to `frontend/`
3. Either edit `PRODUCTION_API_URL` in `frontend/config.js` to your Render URL, **or** add `<meta name="api-base-url" content="https://your-backend.onrender.com">` to each HTML page's `<head>` — no rebuild needed for the latter
4. Deploy

### Docker (self-hosted / any VPS)
```bash
docker build -t mineguard-backend .
docker run -p 5000:5000 --env-file backend/.env mineguard-backend
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Root welcome message |
| `/docs` | GET | Interactive Swagger UI |
| `/health` | GET | Liveness/readiness probe + model metrics |
| `/models` | GET | Full model info (metrics, hyperparameters, dataset) |
| `/predict` | POST | Main RUL prediction (rate-limited, default 60/min/IP) |
| `/history` | GET | Last 10 predictions |
| `/machines` | GET | Machine catalogue (max hours per equipment type) |

## Environment Variables

See `backend/.env.example`. Key ones:

| Variable | Default | Purpose |
|---|---|---|
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS allowlist — **restrict this in production** |
| `RATE_LIMIT_PER_MINUTE` | `60` | Requests/min/IP on `/predict` |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `MODEL_DIR` | *(auto)* | Override if model artifacts live elsewhere (e.g. custom Docker mount) |

## Future Scope

- [ ] Integration with real operational mining-equipment data, if it becomes available
- [ ] LSTM-based deep learning model for genuine time-series RUL prediction (FD001 supports full trajectories, not just snapshots)
- [ ] Multi-equipment fleet monitoring dashboard
- [ ] Digital twin what-if maintenance scheduling
- [ ] SMS/email alerts when RUL drops below a configurable threshold

---

**Built by:** [Swasti Sundar Nath]
**Contact:** [nathswasti.aiml@gmail.com]
