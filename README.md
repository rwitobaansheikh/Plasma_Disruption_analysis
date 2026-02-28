<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/AWS-Fargate-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white" />
</p>

# 🔥 Plasma Disruption Detection Dashboard

A real-time **Bayesian Neural Network** dashboard for detecting plasma disruptions in tokamak fusion reactors, powered by **Monte Carlo Dropout** uncertainty quantification.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Input Features](#input-features)
- [API Endpoints](#api-endpoints)
- [Getting Started](#getting-started)
- [Docker Deployment](#docker-deployment)
- [AWS Fargate Deployment](#aws-fargate-deployment)
- [CI/CD Pipeline](#cicd-pipeline)
- [Project Structure](#project-structure)
- [Contact](#contact)

---

## Overview

Plasma disruptions are sudden, catastrophic losses of plasma confinement in tokamak fusion reactors that can cause severe damage to reactor components. This dashboard provides a **real-time disruption prediction system** using a Bayesian Neural Network that not only predicts disruption probability but also quantifies the **uncertainty** of each prediction — a critical feature for safety-critical decision making.

The model ingests **20 physics-based plasma diagnostic signals** and outputs:
- **Disruption probability** (0–1)
- **Prediction uncertainty** via MC Dropout (50 forward passes)
- **95% confidence intervals**
- **Adaptive risk classification** (HIGH / MEDIUM / LOW)

---

## Features

| Feature | Description |
|---------|-------------|
| 🧠 **Bayesian Neural Network** | Conv1D + Dense layers with MC Dropout for uncertainty estimation |
| 📊 **Interactive Dashboard** | Real-time charts for probability distribution, confidence intervals, and risk levels |
| 🎯 **Uncertainty Quantification** | 50-sample Monte Carlo Dropout produces calibrated confidence intervals |
| 🔬 **Physics-Based Inputs** | 20 tokamak diagnostic signals across 3 plasma regimes (stable, marginal, disruptive) |
| ⚠️ **Adaptive Alert System** | Percentile-based risk thresholds that calibrate per-batch |
| 🐳 **Optimized Docker Image** | Multi-stage build with CPU-only PyTorch (~1.3 GB vs ~8 GB CUDA) |
| 🚀 **AWS Fargate Deployment** | Serverless container hosting with automatic DNS updates |
| 🔄 **CI/CD Pipeline** | GitHub Actions → ECR → ECS → Cloudflare DNS |

---

## Architecture

See the full architecture diagram in [ARCHITECTURE.md](ARCHITECTURE.md).

```
┌─────────────────────────────────────────────────────────┐
│                     User's Browser                       │
│   Chart.js  ·  Plotly.js  ·  Interactive Dashboard       │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP (port 5000)
┌──────────────────────▼──────────────────────────────────┐
│                    Flask Backend                          │
│  /                  → Dashboard UI                       │
│  /api/generate-data → Synthetic tokamak data             │
│  /api/predict       → MC Dropout inference + UQ          │
│  /api/model-info    → Model metadata                     │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│          Bayesian Neural Network (PyTorch)                │
│  Input(20) → Conv1D(16) → Dense(256→128→64) → Sigmoid   │
│  MC Dropout: 50 forward passes → mean ± 1.96σ           │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | HTML5, CSS3, JavaScript, Chart.js, Plotly.js |
| **Backend** | Python 3.11, Flask 3.x |
| **ML Framework** | PyTorch 2.x (CPU) |
| **Data Processing** | NumPy, Pandas, scikit-learn |
| **Containerization** | Docker (multi-stage build) |
| **Cloud Hosting** | AWS Fargate (ECS) |
| **Container Registry** | Amazon ECR Public |
| **CI/CD** | GitHub Actions |
| **DNS** | Cloudflare (proxied) |

---

## Input Features

The model accepts **20 physics-based plasma diagnostic signals**:

| # | Feature | Description | Unit |
|---|---------|-------------|------|
| 1 | `Ip` | Plasma current | MA |
| 2 | `dIp_dt` | Plasma current rate of change | MA/s |
| 3 | `q95` | Safety factor at 95% flux surface | — |
| 4 | `dq_dt` | Safety factor rate of change | 1/s |
| 5 | `li` | Internal inductance | — |
| 6 | `dli_dt` | Internal inductance rate of change | 1/s |
| 7 | `beta` | Normalized beta (plasma pressure) | — |
| 8 | `dbeta_dt` | Beta rate of change | 1/s |
| 9 | `mirnov_dB_dt` | Mirnov coil signal (MHD activity) | T/s |
| 10 | `locked_mode_indicator` | Locked mode amplitude | a.u. |
| 11 | `n1_rms` | n=1 mode RMS amplitude | a.u. |
| 12 | `n2_rms` | n=2 mode RMS amplitude | a.u. |
| 13 | `bolometry` | Radiated power (bolometer) | kW |
| 14 | `Te` | Electron temperature | keV |
| 15 | `dTe_dt` | Electron temperature rate of change | keV/s |
| 16 | `ne_greenwald_frac` | Greenwald density fraction | — |
| 17 | `d2Ip_dt2` | Second derivative of plasma current | MA/s² |
| 18 | `distance_to_wall` | Plasma-wall gap distance | m |
| 19 | `error_field` | Error field magnitude | a.u. |
| 20 | `stability_index` | Composite stability metric | a.u. |

---

## API Endpoints

### `GET /`
Serves the interactive dashboard UI.

### `POST /api/generate-data`
Generates synthetic tokamak plasma data across three regimes.

```json
// Request
{ "num_samples": 100 }

// Response
{
  "status": "success",
  "num_samples": 100,
  "features": ["Ip", "dIp_dt", ...],
  "data": { "Ip": [1.82, ...], ... }
}
```

### `POST /api/predict`
Runs Bayesian inference with MC Dropout uncertainty quantification.

```json
// Response
{
  "status": "success",
  "predictions": [
    {
      "disruption_prob": 0.73,
      "uncertainty": 0.08,
      "confidence": 0.92,
      "lower_ci": 0.57,
      "upper_ci": 0.89,
      "alert": "HIGH"
    }
  ],
  "statistics": {
    "mean_probability": 0.48,
    "mean_uncertainty": 0.06,
    "high_risk_count": 25,
    "medium_risk_count": 30,
    "low_risk_count": 45
  }
}
```

### `GET /api/model-info`
Returns model metadata (name, features, device, UQ method).

---

## Getting Started

### Prerequisites
- Python 3.11+
- pip

### Local Setup

```bash
# Clone the repository
git clone https://github.com/rwitobaansheikh/Fusion_disruption.git
cd Fusion_disruption

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Open `http://localhost:5000` in your browser.

---

## Docker Deployment

The Dockerfile uses a **multi-stage build** to keep the image lightweight (~1.3 GB) by installing CPU-only PyTorch.

```bash
# Build
docker build -t plasma-disruption-dashboard .

# Run
docker run -d -p 5000:5000 --name plasma-dashboard plasma-disruption-dashboard
```

Open `http://localhost:5000`.

---

## AWS Fargate Deployment

The application is deployed as a serverless container on **AWS Fargate**.

### Infrastructure

| Resource | Name |
|----------|------|
| ECS Cluster | `plasma-disruption-cluster` |
| ECS Service | `plasma-disruption-service` |
| Task Definition | `plasma-disruption-task` |
| Container Image | `public.ecr.aws/u6o5d5r2/plasma-disruption-dashboard` |
| Region | `eu-west-2` (London) |
| CPU / Memory | 1 vCPU / 2 GB |

### Push to ECR

```bash
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws/u6o5d5r2
docker tag plasma-disruption-dashboard:latest public.ecr.aws/u6o5d5r2/plasma-disruption-dashboard:latest
docker push public.ecr.aws/u6o5d5r2/plasma-disruption-dashboard:latest
```

---

## CI/CD Pipeline

Every push to `main` triggers the GitHub Actions workflow:

```
git push → GitHub Actions → Build Docker Image → Push to ECR
    → Update ECS Task Definition → Deploy to Fargate
    → Fetch New Public IP → Update Cloudflare DNS
```

### Required GitHub Secrets

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |
| `CF_ZONE_ID` | Cloudflare zone ID |
| `CF_RECORD_ID` | Cloudflare DNS record ID |
| `CLOUDFLARE_API_TOKEN` | Cloudflare API token |

---

## Project Structure

```
Fusion_disruption/
├── app.py                        # Flask backend + model definition + API routes
├── plasma_disruption_model.pt    # Trained PyTorch model weights
├── feature_scaler.pkl            # Fitted StandardScaler for input normalization
├── feature_columns.pkl           # Ordered list of 20 feature names
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Multi-stage Docker build (CPU-only PyTorch)
├── task-def.json                 # ECS Fargate task definition
├── .dockerignore                 # Docker build exclusions
├── .github/
│   └── workflow/
│       └── deploy.yml            # CI/CD: GitHub Actions → ECR → ECS → Cloudflare
├── templates/
│   └── index.html                # Dashboard HTML (Jinja2 template)
├── static/
│   ├── style.css                 # Dashboard styling
│   └── script.js                 # Frontend logic (Chart.js, Plotly, API calls)
├── README.md                     # This file
└── ARCHITECTURE.md               # Visual architecture documentation
```

---

## Contact

**Rwitobaan Sheikh**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/rwitobaansheikh)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/rwitobaansheikh)
[![Email](https://img.shields.io/badge/Email-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:rwitobaansheikh@gmail.com)

---

<p align="center">
  <sub>Built with 🔥 for fusion energy research</sub>
</p>
