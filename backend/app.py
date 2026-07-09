"""
MineGuard RUL Prediction API — v5.1
Backed by a model trained on NASA C-MAPSS FD001 (real turbofan run-to-failure data),
applied to 6 mining machine types via a real-world hours scaling layer.

Endpoints:
  GET  /          — root welcome message
  GET  /health    — liveness/readiness probe + model status + metrics
  GET  /models    — full model info for About page
  GET  /history   — last 10 predictions
  POST /predict   — main RUL prediction endpoint (rate-limited)

Why FD001 instead of a "mining" dataset:
  Real mining-equipment run-to-failure data is proprietary and not publicly released.
  FD001 gives a genuinely OBSERVED remaining-useful-life label (cycles until an engine
  actually failed in NASA's simulation) rather than a formula we invented — see
  model/train_model.ipynb for the full reasoning and literature citations.

Machine types and their normal-condition max RUL hours (scaling layer only —
machine type is NOT a model input, it only rescales the model's cycle output into
a plausible real-world hours figure per equipment class):
  Scraper : 25,000 hrs (Light)
  Grader  : 30,000 hrs (Light)
  Dozer   : 40,000 hrs (Medium)
  Shovel  : 50,000 hrs (Medium)
  Dumper  : 40,000 hrs (Heavy)
  Dragline: 120,000 hrs (Heavy)

Scaling logic:
  1. Model predicts raw_rul_cycles in [0, 125]  (FD001's standard clipped ceiling)
  2. health_pct = raw_rul_cycles / 125          (fraction of life remaining, 0-1)
  3. rul_hours  = health_pct x machine_max      (real-world hours remaining)
  4. Risk thresholds are applied on health_pct, not raw cycles

Configuration is environment-driven — see config.py and .env.example.

Run locally:
  uvicorn app_v4:app --reload --port 5000

Run in production (e.g. Render/Railway):
  uvicorn app_v4:app --host 0.0.0.0 --port $PORT --workers 2
"""

import logging
import os
import json
import csv
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from config import settings

# ==============================================================================
# Logging
# ==============================================================================
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("mineguard")

# ==============================================================================
# Paths
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = settings.MODEL_DIR or os.path.join(BASE_DIR, "..", "model")
LOG_PATH = os.path.join(BASE_DIR, "predictions.log")

MODEL_PATH = os.path.join(MODEL_DIR, "rul_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.pkl")
EXPLAINER_PATH = os.path.join(MODEL_DIR, "shap_explainer.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

# Mutable container populated at startup by the lifespan handler below.
# Kept as a plain dict (rather than globals reassigned deep in a function) so
# it's obvious everything the request handlers read comes from one place.
ml: dict = {}
_log_lock = threading.Lock()  # predictions.log is written from request handlers;
                               # a lock keeps concurrent writes from interleaving.

# ==============================================================================
# Machine catalogue (real-world hours scaling only — NOT a model feature)
# ==============================================================================
MACHINE_CATALOGUE = {
    "Scraper": {"grade": "L", "label": "Light", "max_hours": 25000, "icon": "🔧"},
    "Grader": {"grade": "L", "label": "Light", "max_hours": 30000, "icon": "🛤"},
    "Dozer": {"grade": "M", "label": "Medium", "max_hours": 40000, "icon": "🚜"},
    "Shovel": {"grade": "M", "label": "Medium", "max_hours": 50000, "icon": "🏗"},
    "Dumper": {"grade": "H", "label": "Heavy", "max_hours": 40000, "icon": "🚛"},
    "Dragline": {"grade": "H", "label": "Heavy", "max_hours": 120000, "icon": "⚙️"},
}

# Friendly labels for each FD001 sensor — used for SHAP failure-reason text and docs.
FEATURE_LABELS = {
    "sensor_2": "LPC outlet temperature rising (T24)",
    "sensor_3": "HPC outlet temperature rising (T30)",
    "sensor_4": "LPT outlet temperature rising (T50)",
    "sensor_7": "HPC outlet pressure shifting (P30)",
    "sensor_8": "Physical fan speed drifting (Nf)",
    "sensor_9": "Physical core speed drifting (Nc)",
    "sensor_11": "HPC static pressure shifting (Ps30)",
    "sensor_12": "Fuel-flow / pressure ratio shifting (phi)",
    "sensor_13": "Corrected fan speed drifting (NRf)",
    "sensor_14": "Corrected core speed drifting (NRc)",
    "sensor_15": "Bypass ratio shifting (BPR)",
    "sensor_17": "Bleed enthalpy shifting (htBleed)",
    "sensor_20": "HPT coolant bleed shifting (W31)",
    "sensor_21": "LPT coolant bleed shifting (W32)",
}

# Field bounds derived from the real FD001 training distribution (min/max with a
# small safety margin) — NOT arbitrary guesses. See model/train_model.ipynb Step 4.
SENSOR_BOUNDS = {
    "sensor_2": (640.0, 646.0),
    "sensor_3": (1568.0, 1620.0),
    "sensor_4": (1378.0, 1445.0),
    "sensor_7": (548.0, 558.0),
    "sensor_8": (2387.5, 2389.0),
    "sensor_9": (9000.0, 9260.0),
    "sensor_11": (46.5, 49.0),
    "sensor_12": (517.0, 525.0),
    "sensor_13": (2387.5, 2389.0),
    "sensor_14": (8080.0, 8310.0),
    "sensor_15": (8.2, 8.7),
    "sensor_17": (386.0, 402.0),
    "sensor_20": (37.9, 39.7),
    "sensor_21": (22.7, 23.8),
}


# ==============================================================================
# Lifespan — load model artifacts once at startup, fail loudly and clearly if missing
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    required = {
        "model": MODEL_PATH,
        "scaler": SCALER_PATH,
        "explainer": EXPLAINER_PATH,
        "metadata": METADATA_PATH,
    }
    missing = [name for name, path in required.items() if not os.path.isfile(path)]
    if missing:
        raise RuntimeError(
            f"Missing model artifact(s): {missing}. Expected them under {MODEL_DIR}. "
            f"Run model/train_model.ipynb first, or set MODEL_DIR to the correct path."
        )

    logger.info("Loading model artifacts from %s ...", MODEL_DIR)
    ml["model"] = joblib.load(MODEL_PATH)
    ml["scaler"] = joblib.load(SCALER_PATH)
    ml["explainer"] = joblib.load(EXPLAINER_PATH)
    with open(METADATA_PATH) as f:
        ml["metadata"] = json.load(f)

    ml["features"] = ml["metadata"]["features"]
    ml["rul_clip"] = ml["metadata"].get("rul_clip", 125.0)

    logger.info("Model    : %s", ml["metadata"].get("model_name"))
    logger.info("Test R2  : %s", ml["metadata"].get("metrics", {}).get("test_r2", "N/A"))
    logger.info("Features : %s", ml["features"])
    logger.info("CORS origins: %s", settings.cors_origins_list)
    logger.info("Rate limit  : %s requests/minute on /predict", settings.RATE_LIMIT_PER_MINUTE)

    yield  # ---- app runs here ----

    logger.info("Shutting down MineGuard API.")
    ml.clear()


# ==============================================================================
# App
# ==============================================================================
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="MineGuard RUL Prediction API",
    description=(
        "Predicts Remaining Useful Life (RUL) of open-pit mining equipment. "
        "The underlying model is trained on NASA C-MAPSS FD001 (real turbofan "
        "run-to-failure data) since no public real-world mining-equipment RUL "
        "dataset exists; sensor readings are used as illustrative analogs for "
        "heavy-machinery degradation. RUL is scaled per machine type to "
        "real-world operating hours. SHAP provides per-sample failure explanations."
    ),
    version="5.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all so unexpected errors return a clean JSON 500 instead of leaking
    stack traces to the client. The real exception is still logged server-side."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check server logs for details."},
    )


# ==============================================================================
# Schemas
# ==============================================================================
MachineType = Literal["Scraper", "Grader", "Dozer", "Shovel", "Dumper", "Dragline"]


class PredictRequest(BaseModel):
    machine_type: MachineType = Field(
        default="Dozer",
        description="Mining machine type. Used only to rescale predicted cycles into real-world hours for that equipment class.",
    )
    sensor_2: float = Field(..., ge=SENSOR_BOUNDS["sensor_2"][0], le=SENSOR_BOUNDS["sensor_2"][1],
        description="LPC outlet temperature (T24), degR. Typical: 641.9-643.6")
    sensor_3: float = Field(..., ge=SENSOR_BOUNDS["sensor_3"][0], le=SENSOR_BOUNDS["sensor_3"][1],
        description="HPC outlet temperature (T30), degR. Typical: 1581.1-1601.5")
    sensor_4: float = Field(..., ge=SENSOR_BOUNDS["sensor_4"][0], le=SENSOR_BOUNDS["sensor_4"][1],
        description="LPT outlet temperature (T50), degR. Typical: 1395.6-1425.7")
    sensor_7: float = Field(..., ge=SENSOR_BOUNDS["sensor_7"][0], le=SENSOR_BOUNDS["sensor_7"][1],
        description="HPC outlet pressure (P30), psia. Typical: 551.7-554.7")
    sensor_8: float = Field(..., ge=SENSOR_BOUNDS["sensor_8"][0], le=SENSOR_BOUNDS["sensor_8"][1],
        description="Physical fan speed (Nf), rpm. Typical: 2388.0-2388.2")
    sensor_9: float = Field(..., ge=SENSOR_BOUNDS["sensor_9"][0], le=SENSOR_BOUNDS["sensor_9"][1],
        description="Physical core speed (Nc), rpm. Typical: 9042.6-9110.0")
    sensor_11: float = Field(..., ge=SENSOR_BOUNDS["sensor_11"][0], le=SENSOR_BOUNDS["sensor_11"][1],
        description="HPC static pressure (Ps30), psia. Typical: 47.15-48.05")
    sensor_12: float = Field(..., ge=SENSOR_BOUNDS["sensor_12"][0], le=SENSOR_BOUNDS["sensor_12"][1],
        description="Fuel flow / static pressure ratio (phi), pps/psi. Typical: 520.0-522.5")
    sensor_13: float = Field(..., ge=SENSOR_BOUNDS["sensor_13"][0], le=SENSOR_BOUNDS["sensor_13"][1],
        description="Corrected fan speed (NRf), rpm. Typical: 2388.0-2388.2")
    sensor_14: float = Field(..., ge=SENSOR_BOUNDS["sensor_14"][0], le=SENSOR_BOUNDS["sensor_14"][1],
        description="Corrected core speed (NRc), rpm. Typical: 8122.5-8181.4")
    sensor_15: float = Field(..., ge=SENSOR_BOUNDS["sensor_15"][0], le=SENSOR_BOUNDS["sensor_15"][1],
        description="Bypass ratio (BPR). Typical: 8.386-8.511")
    sensor_17: float = Field(..., ge=SENSOR_BOUNDS["sensor_17"][0], le=SENSOR_BOUNDS["sensor_17"][1],
        description="Bleed enthalpy (htBleed). Typical: 391-396")
    sensor_20: float = Field(..., ge=SENSOR_BOUNDS["sensor_20"][0], le=SENSOR_BOUNDS["sensor_20"][1],
        description="HPT coolant bleed (W31), lbm/s. Typical: 38.49-39.09")
    sensor_21: float = Field(..., ge=SENSOR_BOUNDS["sensor_21"][0], le=SENSOR_BOUNDS["sensor_21"][1],
        description="LPT coolant bleed (W32), lbm/s. Typical: 23.09-23.45")

    @model_validator(mode="after")
    def lpt_must_exceed_lpc(self):
        # Genuine thermodynamic constraint (not an invented rule): in a turbofan,
        # the low-pressure turbine outlet (T50) is always hotter than the
        # low-pressure compressor outlet (T24) -- confirmed directly in the FD001
        # data (T24 ~641-644 degR vs T50 ~1380-1441 degR).
        if self.sensor_4 <= self.sensor_2:
            raise ValueError(
                f"sensor_4 / T50 ({self.sensor_4} degR) must exceed sensor_2 / T24 ({self.sensor_2} degR) "
                f"-- the turbine outlet is always hotter than the compressor outlet."
            )
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "machine_type": "Dozer",
                "sensor_2": 642.7, "sensor_3": 1590.5, "sensor_4": 1408.9,
                "sensor_7": 553.4, "sensor_8": 2388.1, "sensor_9": 9065.2,
                "sensor_11": 47.54, "sensor_12": 521.4, "sensor_13": 2388.1,
                "sensor_14": 8143.8, "sensor_15": 8.44, "sensor_17": 393,
                "sensor_20": 38.82, "sensor_21": 23.29,
            }]
        }
    }


class FailureReason(BaseModel):
    reason: str
    contribution: float
    importance: float
    direction: str
    direction_icon: str


class ComputedValues(BaseModel):
    temp_rise: float    # sensor_4 (T50) - sensor_2 (T24), degR -- overall temperature rise across the engine
    speed_ratio: float  # sensor_9 (Nc) / sensor_8 (Nf) -- core-to-fan speed ratio


class MachineInfo(BaseModel):
    type: str
    grade: str
    grade_label: str
    max_hours: int
    icon: str


class PredictResponse(BaseModel):
    raw_rul_cycles: float           # raw model output, 0-125 cycles (FD001 clipped scale)
    health_pct: float               # 0-100, fraction of life remaining
    rul_hours: float                # = health_pct/100 x machine max_hours
    rul_hours_label: str            # human-friendly e.g. "12,450 hrs"
    machine: MachineInfo
    risk_level: Literal["low", "medium", "high", "critical"]
    health_score: float             # same as health_pct, kept for frontend compat
    failure_reasons: list[FailureReason]
    computed: ComputedValues
    model: str


class HealthResponse(BaseModel):
    status: str
    model: str
    metrics: dict
    cv_r2: str


class ModelsResponse(BaseModel):
    best_model: str
    metrics: dict
    cv_r2_mean: Optional[float] = None
    cv_r2_std: Optional[float] = None
    n_features: int
    dataset: str
    dataset_note: str
    machines: dict


class HistoryResponse(BaseModel):
    predictions: list[dict]
    total: int


# ==============================================================================
# Helpers
# ==============================================================================
def build_feature_row(req: PredictRequest) -> pd.DataFrame:
    """Assemble the 14 raw sensor readings in the exact order the scaler/model expect."""
    row = {f: getattr(req, f) for f in ml["features"]}
    return pd.DataFrame([row])[ml["features"]]


def compute_display_values(req: PredictRequest) -> ComputedValues:
    """Purely informational -- NOT fed into the model. Real physical relationships
    (temperature rise across the engine, core-to-fan speed ratio), shown to the
    user for context alongside the SHAP-driven failure reasons."""
    return ComputedValues(
        temp_rise=round(req.sensor_4 - req.sensor_2, 2),
        speed_ratio=round(req.sensor_9 / req.sensor_8, 4),
    )


def log_prediction(req: PredictRequest, raw_rul: float, scaled_rul: float,
                    health_pct: float, risk: str) -> None:
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "machine_type": req.machine_type,
        "grade": MACHINE_CATALOGUE[req.machine_type]["grade"],
        "raw_rul_cycles": raw_rul,
        "health_pct": health_pct,
        "rul_scaled_hrs": scaled_rul,
        "risk_level": risk,
    }
    try:
        with _log_lock:
            exists = os.path.isfile(LOG_PATH)
            with open(LOG_PATH, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
    except OSError as e:
        # Logging predictions is best-effort — a full disk or read-only filesystem
        # (common on some free-tier hosts) should never break a live prediction.
        logger.warning("Could not write predictions.log: %s", e)


# ==============================================================================
# Routes
# ==============================================================================
@app.get("/", tags=["Root"])
def root():
    return {
        "message": "MineGuard RUL Prediction API v5 is running.",
        "docs": "/docs",
        "predict": "POST /predict",
        "machines": list(MACHINE_CATALOGUE.keys()),
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
def predict(request: Request, req: PredictRequest) -> PredictResponse:
    try:
        X = build_feature_row(req)
        X_scaled = ml["scaler"].transform(X)

        # 1. Raw model prediction (0-125 clipped cycle scale)
        raw_rul = float(ml["model"].predict(X_scaled)[0])
        raw_rul = max(0.0, min(ml["rul_clip"], round(raw_rul, 2)))

        # 2. Health percentage (fraction of clipped life remaining)
        health_pct = round(min(100.0, (raw_rul / ml["rul_clip"]) * 100), 2)

        # 3. Scale to real-world hours for this machine
        cat = MACHINE_CATALOGUE[req.machine_type]
        max_hours = cat["max_hours"]
        scaled_rul = round((health_pct / 100.0) * max_hours, 0)

        # 4. Risk level -- based on health_pct (universal across machine types)
        if health_pct == 0:
            risk = "critical"
        elif health_pct < 20:
            risk = "critical"
        elif health_pct < 40:
            risk = "high"
        elif health_pct < 65:
            risk = "medium"
        else:
            risk = "low"

        # 5. SHAP failure reasons -- explainer expects SCALED input, same as training
        shap_vals = ml["explainer"].shap_values(X_scaled)[0]
        abs_c = np.abs(shap_vals)
        ranked_idx = np.argsort(abs_c)[::-1][:3]

        top3 = []
        for i in ranked_idx:
            contrib = round(float(abs_c[i]), 2)
            top3.append(FailureReason(
                reason=FEATURE_LABELS[ml["features"][i]],
                contribution=contrib,
                importance=contrib,
                direction="reduces RUL" if shap_vals[i] < 0 else "extends RUL",
                direction_icon="↓" if shap_vals[i] < 0 else "↑",
            ))

        log_prediction(req, raw_rul, scaled_rul, health_pct, risk)

        return PredictResponse(
            raw_rul_cycles=raw_rul,
            health_pct=health_pct,
            rul_hours=scaled_rul,
            rul_hours_label=f"{int(scaled_rul):,} hrs",
            machine=MachineInfo(
                type=req.machine_type,
                grade=cat["grade"],
                grade_label=cat["label"],
                max_hours=max_hours,
                icon=cat["icon"],
            ),
            risk_level=risk,
            health_score=health_pct,
            failure_reasons=top3,
            computed=compute_display_values(req),
            model=ml["metadata"].get("model_name", "RandomForest"),
        )

    except (KeyError, ValueError) as e:
        # Known, expected failure modes (bad feature name, bad numeric input) get
        # a clean 400 instead of falling through to the generic 500 handler.
        logger.warning("Bad prediction request: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health", response_model=HealthResponse, tags=["Status"])
def health():
    """Liveness/readiness probe. Returns 200 with model status whenever the
    process is up and the model is loaded — suitable for Render/Railway/Docker
    health checks."""
    metadata = ml.get("metadata", {})
    return HealthResponse(
        status="ok" if ml.get("model") is not None else "model_not_loaded",
        model=metadata.get("model_name", "unknown"),
        metrics=metadata.get("metrics", {}),
        cv_r2=f"{metadata.get('cv_r2_mean', 'N/A')} +/- {metadata.get('cv_r2_std', 'N/A')}",
    )


@app.get("/models", response_model=ModelsResponse, tags=["Status"])
def models_info():
    metadata = ml.get("metadata", {})
    return ModelsResponse(
        best_model=metadata.get("model_name", ""),
        metrics=metadata.get("metrics", {}),
        cv_r2_mean=metadata.get("cv_r2_mean"),
        cv_r2_std=metadata.get("cv_r2_std"),
        n_features=len(ml.get("features", [])),
        dataset=metadata.get("dataset", "NASA C-MAPSS FD001"),
        dataset_note=metadata.get("dataset_note", ""),
        machines={k: v for k, v in MACHINE_CATALOGUE.items()},
    )


@app.get("/history", response_model=HistoryResponse, tags=["History"])
def history():
    if not os.path.isfile(LOG_PATH):
        return HistoryResponse(predictions=[], total=0)
    try:
        df_log = pd.read_csv(LOG_PATH)
        last10 = df_log.tail(10).iloc[::-1].to_dict(orient="records")
        return HistoryResponse(predictions=last10, total=len(df_log))
    except (OSError, pd.errors.ParserError) as e:
        logger.warning("Could not read predictions.log: %s", e)
        return HistoryResponse(predictions=[], total=0)


@app.get("/machines", tags=["Status"])
def machines():
    """Returns the full machine catalogue with max hours for frontend use."""
    return MACHINE_CATALOGUE
