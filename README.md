# AI-Generated Text Detection

Multilingual AI-generated text analysis with confidence-aware predictions.

> **AI-text detection is probabilistic and should not be treated as definitive proof of authorship.** Human and AI-generated writing distributions overlap — this tool is a decision-support signal, not a verdict.

## Overview

AI-Generated Text Detection is a full-stack web application that analyzes a piece of writing (e.g. a student submission) and classifies it as:

- **HUMAN-WRITTEN**
- **AI-GENERATED**
- **UNCERTAIN** (the system abstains rather than guessing when evidence is weak)

It supports **English, Telugu, and Hindi**, with automatic language detection, and returns a calibrated confidence score alongside a breakdown of the evidence (stylometry, pseudo-perplexity, curvature) that produced the result.

**Live Demo:** Add deployed URL here

## Features

- Multilingual analysis (English / Telugu / Hindi, auto-detect or manual selection)
- Combined stylometric + language-model-predictability + curvature feature pipeline
- Calibrated probabilities (not raw classifier scores) via `CalibratedClassifierCV`
- Calibrated abstention: the model says UNCERTAIN instead of forcing a low-confidence guess
- Segment-level predictability chart
- REST API (FastAPI) with input validation, structured errors, and a `/health` endpoint
- React + Vite frontend, responsive, no unnecessary dependencies
- Privacy-conscious: submitted text is never permanently stored or logged
- Ready to deploy: Vercel (frontend) + Render (backend), Docker-based

## Architecture

```
React (Vite) frontend  --HTTPS-->  FastAPI backend  -->  ML pipeline (scikit-learn + XLM-RoBERTa)
```

A deliberately simple two-tier architecture: no microservices, no message queue, no database. The backend is stateless — every request is analyzed independently and nothing about the submission is persisted.

## Technologies

| Layer | Technology |
|---|---|
| Frontend | React, Vite, JavaScript |
| Backend | Python, FastAPI, Uvicorn |
| ML | scikit-learn, numpy, pandas, joblib, scipy |
| NLP | Hugging Face Transformers, PyTorch (XLM-RoBERTa) |
| Deployment | Vercel (frontend), Render (backend), Docker |

## Folder Structure

```
ai-generated-text-detection/
├── frontend/            React + Vite app
│   ├── src/components/  Header, TextInput, AnalysisResult, ConfidenceMeter,
│   │                     FeatureCard, PerplexityChart, Footer
│   ├── src/pages/        Home, About
│   └── src/services/     api.js (backend client)
├── backend/              FastAPI app
│   ├── app/api/           routes.py  (/api/analyze, health helper)
│   ├── app/core/          config.py, logging_config.py
│   ├── app/preprocessing/ text_cleaner.py, language_detector.py
│   ├── app/features/      stylometry.py, perplexity.py, curvature.py, feature_extractor.py
│   ├── app/ml/            model_loader.py, predictor.py, calibration.py
│   ├── app/schemas/       prediction.py (Pydantic request/response models)
│   └── models/            trained model artifacts (generated, not committed)
├── training/              Dataset prep, feature extraction, training, calibration, evaluation, experiments
├── tests/                 pytest suite
├── .gitignore
├── README.md
└── LICENSE
```

## How It Works

A submission goes through four stages:

1. **Preprocessing** — Unicode normalization, sentence/paragraph splitting (aware of Devanagari/Telugu sentence-final punctuation `।` and `॥`), and language detection.
2. **Feature extraction** — stylometric, pseudo-perplexity, and curvature features are computed (see below).
3. **Classification + calibration** — a trained classifier produces a raw score, which is passed through a calibration layer so the reported number is an actual probability, not an arbitrary score.
4. **Abstention** — if the calibrated confidence is below a validated threshold, the system reports UNCERTAIN instead of forcing HUMAN-WRITTEN or AI-GENERATED.

### Stylometric Analysis

Unicode-aware statistics: word/character/sentence/paragraph counts, average word and sentence length, sentence-length variance/std-dev, vocabulary size and type-token ratio, punctuation frequencies (comma, period, question mark, exclamation, semicolon, colon), digit/uppercase/lowercase/whitespace ratios, and short-/long-word ratios. See `backend/app/features/stylometry.py`.

### Pseudo-Perplexity

XLM-RoBERTa is a **masked language model**, not a causal/autoregressive model like GPT. Standard autoregressive perplexity — which conditions each token only on preceding tokens — is not defined for a masked LM. Instead, this project computes **pseudo-perplexity** (Salazar et al., 2020): each token is masked one at a time, the model's probability for the true token is recorded, and the results are averaged and exponentiated. The application and API always call this "pseudo-perplexity", never "perplexity", to avoid a false scientific claim. See `backend/app/features/perplexity.py`.

Low pseudo-perplexity (highly predictable text) is **not**, by itself, proof of AI generation — very formulaic human writing can also score low. It is one signal among several fed into a calibrated classifier.

### Curvature Analysis

The text is split into segments, and pseudo-perplexity is computed per segment, producing a sequence `P = [P_1 .. P_n]`. First differences (`ΔP_i = P_(i+1) - P_i`) and second differences (`Δ²P_i = P_(i+1) - 2P_i + P_(i-1)`) are computed, and summary statistics (mean, std, mean-absolute, max-absolute, sign changes) are extracted. Curvature alone does not prove AI generation — like pseudo-perplexity, it is a weak, probabilistic signal combined with everything else. See `backend/app/features/curvature.py`.

### Machine Learning

Logistic Regression and Random Forest are trained and compared on a stratified 70/15/15 train/validation/test split (XGBoost is used automatically if installed). The best model by validation ROC-AUC is selected. See `training/train.py`.

### Calibration

The selected model's probabilities are calibrated with `CalibratedClassifierCV` — sigmoid (Platt scaling) for smaller validation sets, isotonic for larger ones. The API never reports a raw, uncalibrated classifier score as "confidence". See `training/calibrate.py` and `backend/app/ml/calibration.py`.

### Abstention

An abstention threshold is chosen by sweeping candidate thresholds on the validation set and evaluating **coverage** (fraction of predictions made), **selective risk** (error rate among predictions made), and **accuracy on accepted predictions** — not chosen arbitrarily. Below the threshold, the API returns UNCERTAIN. See `training/calibrate.py`.

## Dataset

**No real dataset ships with this repository.** A legitimate detector must be trained on genuinely human-written and genuinely AI-generated text that you collect and have the rights to use. See `training/data/README.md` for the expected folder structure and guidance on building a real dataset (avoiding topic leakage and duplicated texts across splits).

A tiny synthetic dataset is available for pipeline smoke-testing only:

```bash
python training/prepare_dataset.py --demo
```

**DEMO DATA — NOT SUITABLE FOR RESEARCH EVALUATION.** Any metrics produced from it are meaningless and must not be reported as real results.

## Model Training

From the `training/` directory (with `training/requirements.txt` installed):

```bash
python prepare_dataset.py          # build manifest.csv from training/data/ (or --demo for a smoke test)
python extract_features.py         # extract stylometry + pseudo-perplexity + curvature -> features.csv
python train.py                    # train + select best classifier -> best_model_raw.joblib
python calibrate.py                # calibrate + choose abstention threshold -> backend/models/
python evaluate.py                 # accuracy/precision/recall/F1/ROC-AUC/coverage/selective risk
python experiments.py              # ablation comparison -> model_comparison.csv + plot
```

`calibrate.py` writes `backend/models/calibrated_model.joblib` and `backend/models/model_metadata.json`, which the API loads at runtime. No numbers from these scripts are pre-filled in this README — run them yourself and report what you actually measure.

## Local Development

### Backend (Windows / VS Code)

```bat
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000` (docs at `/docs`). Copy `backend/.env.example` to `backend/.env` and adjust as needed — `CORS_ORIGINS` must include your frontend's local URL (default `http://localhost:5173`).

### Frontend (Windows / VS Code)

```bat
cd frontend
npm install
npm run dev
```

Copy `frontend/.env.example` to `frontend/.env` and set:

```
VITE_API_URL=http://localhost:8000
```

The app will be available at `http://localhost:5173`.

## Backend Deployment (Render)

1. Push this repository to GitHub.
2. In Render, create a new **Web Service** from the repo, using `backend/render.yaml` (Blueprint) or manually pointing `dockerfilePath`/`dockerContext` at `backend/`.
3. Set environment variables (see `backend/.env.example`) — at minimum `CORS_ORIGINS` set to your Vercel frontend URL.
4. Render provides `$PORT` automatically; the Dockerfile's `CMD` already binds to `0.0.0.0:${PORT}`.
5. Confirm `GET /health` returns `200` once deployed.

## Frontend Deployment (Vercel)

1. Import the repository into Vercel, set the project root to `frontend/`.
2. Vercel will use `frontend/vercel.json` for build settings (`npm run build`, output `dist/`).
3. Set the environment variable `VITE_API_URL` to your deployed Render backend URL.
4. Deploy, then update the backend's `CORS_ORIGINS` to include the resulting Vercel URL, and redeploy the backend.

## Environment Variables

**Backend** (`backend/.env.example`): `CORS_ORIGINS`, `MAX_TEXT_LENGTH`, `MIN_TEXT_LENGTH`, `PERPLEXITY_MODEL_NAME`, `INFERENCE_DEVICE`, `MODEL_DIR`, `PORT`.

**Frontend** (`frontend/.env.example`): `VITE_API_URL`.

No production URLs are hard-coded anywhere in the codebase.

## API Documentation

Interactive OpenAPI docs are available at `/docs` on the running backend. Summary:

### `POST /api/analyze`

```json
// Request
{ "text": "...", "language": "auto" }

// Response
{
  "prediction": "AI-GENERATED",
  "confidence": 0.91,
  "abstained": false,
  "language": "English",
  "word_count": 420,
  "sentence_count": 22,
  "stylometric_features": { "...": "..." },
  "pseudo_perplexity": { "mean": 24.5, "median": 23.1, "std": 4.2 },
  "curvature": { "mean_first_difference": 1.2, "mean_second_difference": -0.4, "max_absolute_curvature": 5.2 },
  "segments": [{ "segment": 1, "score": 21.3 }],
  "explanation": "...",
  "model_version": "v20260101-120000",
  "disclaimer": "AI-text detection is probabilistic and should not be treated as definitive proof of authorship."
}
```

Validation errors return `422` with a plain-language `detail` message; an untrained/missing model returns `503`; unexpected errors return a sanitized `500` (no stack traces are exposed to clients).

### `GET /health`

```json
{ "status": "healthy", "model_loaded": true, "model_version": "v20260101-120000", "supported_languages": ["en", "te", "hi"] }
```

## Testing

```bash
pip install -r backend/requirements.txt pytest httpx
pytest tests/
```

Covers stylometric feature extraction, curvature calculations, text-length validation, calibration/abstention logic, request-schema validation, `/health`, and `/api/analyze` (the full-response schema test is skipped automatically if no trained model is present).

## Limitations

- AI-text detection is inherently probabilistic; there is no such thing as 100% reliable detection, and this project makes no such claim.
- Short texts (well under the recommended minimum length) produce noisier, less reliable pseudo-perplexity and curvature signals.
- The system has not been evaluated against adversarial paraphrasing/humanizing tools.
- Performance depends entirely on the quality and size of the dataset you train on — no dataset ships with this repository, and no accuracy numbers are claimed here.
- CPU inference for pseudo-perplexity is inherently slower than GPU inference; for a class-sized deployment this is usually acceptable, but very long submissions will take longer to analyze.

## Privacy

Submitted text is processed in memory to produce a result and is **not** stored by default. Server logs contain only metadata (timestamp, processing duration, language, model version, prediction) — never the submitted text. See `backend/app/api/routes.py` and `backend/app/core/logging_config.py`.

## Ethical Considerations

This tool is intended to support, not replace, a human reviewer's judgment (e.g. a faculty member investigating a potential academic-integrity concern). Given the probabilistic nature of AI-text detection and the risk of false positives against certain writing styles (including non-native English writers, whose text can score differently on predictability-based signals), no automated output from this system should be treated as sole, sufficient evidence for an academic-integrity decision.

## Future Improvements

- Expand language support beyond English, Telugu, and Hindi.
- Add explainability visualizations at the token/sentence level.
- Evaluate robustness against paraphrasing and AI-text "humanizing" tools.
- Support batch/file upload analysis for a full assignment set.
- Explore ensembling multiple language models for pseudo-perplexity.

## Screenshots

_Add screenshots of the running application here._

## How to Explain This in a Viva

See the walkthrough at the end of the build process / project notes for a plain-language explanation of each component, suitable for a B.Tech viva.
