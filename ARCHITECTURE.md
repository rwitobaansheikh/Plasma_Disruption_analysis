# 🏗️ ARCHITECTURE — Plasma Disruption Detection Dashboard

This document presents the end-to-end architecture of the application, including frontend, backend APIs, ML inference pipeline, and cloud deployment.

---

## 1) High-Level System Architecture

```mermaid
%%{init: {
  "theme": "base",
  "themeVariables": {
    "background": "#0B1020",
    "primaryColor": "#1E293B",
    "primaryTextColor": "#F8FAFC",
    "primaryBorderColor": "#38BDF8",
    "lineColor": "#F8FAFC",
    "secondaryColor": "#0F172A",
    "tertiaryColor": "#111827"
  }
}}%%
flowchart LR
    U[👤 User Browser\nDashboard UI\nHTML/CSS/JS + Chart.js + Plotly] -->|HTTP :5000| F[🐍 Flask App\napp.py]

    F --> R1[/GET /\nRender index.html/]
    F --> R2[/POST /api/generate-data\nSynthetic Tokamak Samples/]
    F --> R3[/POST /api/predict\nMC Dropout Inference/]
    F --> R4[/GET /api/model-info\nModel Metadata/]

    R3 --> P[🧠 PyTorch Bayesian Model\nAdvancedBayesianDisruptionModel]
    P --> M[(plasma_disruption_model.pt)]
    R3 --> S[(feature_scaler.pkl)]
    F --> T[templates/index.html]
    F --> ST[static/style.css + static/script.js]

    classDef accent fill:#0F172A,stroke:#22D3EE,stroke-width:2px,color:#F8FAFC;
    classDef file fill:#111827,stroke:#A78BFA,stroke-width:2px,color:#F8FAFC;

    class U,F,P accent;
    class M,S,T,ST file;
```

---

## 2) Inference & Uncertainty Flow

```mermaid
%%{init: {
  "theme": "base",
  "themeVariables": {
    "background": "#0B1020",
    "primaryColor": "#1E293B",
    "primaryTextColor": "#F8FAFC",
    "primaryBorderColor": "#34D399",
    "lineColor": "#F8FAFC"
  }
}}%%
flowchart TD
    A[Input JSON data\n20 plasma features] --> B[Validate & align columns]
    B --> C[Scale with StandardScaler]
    C --> D[Convert to Torch Tensor]
    D --> E[MC Dropout\n50 forward passes]
    E --> F[Mean probability μ]
    E --> G[Uncertainty σ]
    F --> H[95% CI = μ ± 1.96σ]
    G --> H
    H --> I[Adaptive Risk Bands\nLOW / MEDIUM / HIGH]
    I --> J[Response JSON\npredictions + statistics]

    classDef step fill:#111827,stroke:#34D399,stroke-width:2px,color:#F8FAFC;
    class A,B,C,D,E,F,G,H,I,J step;
```

---

## 3) Model Architecture (Conceptual)

```mermaid
%%{init: {
  "theme": "base",
  "themeVariables": {
    "background": "#0B1020",
    "primaryColor": "#1E293B",
    "primaryTextColor": "#F8FAFC",
    "primaryBorderColor": "#F59E0B",
    "lineColor": "#F8FAFC"
  }
}}%%
flowchart LR
    IN[Input\n20 features] --> C1[Conv1D path\n1→16, k=3]
    IN --> SKIP[Skip path\nRaw features]
    C1 --> FLAT[Flatten\n16×32 = 512]
    FLAT --> CAT[Concatenate\n512 + 20]
    SKIP --> CAT
    CAT --> D1[Dense 256 + BN + ReLU + Dropout]
    D1 --> D2[Dense 128 + BN + ReLU + Dropout]
    D2 --> D3[Dense 64 + BN + ReLU + Dropout]
    D3 --> OUT[Sigmoid\nDisruption Probability]

    classDef mdl fill:#111827,stroke:#F59E0B,stroke-width:2px,color:#F8FAFC;
    class IN,C1,SKIP,FLAT,CAT,D1,D2,D3,OUT mdl;
```

---

## 4) Deployment Architecture (AWS)

```mermaid
%%{init: {
  "theme": "base",
  "themeVariables": {
    "background": "#0B1020",
    "primaryColor": "#1E293B",
    "primaryTextColor": "#F8FAFC",
    "primaryBorderColor": "#60A5FA",
    "lineColor": "#F8FAFC"
  }
}}%%
flowchart LR
    GH[GitHub Actions\n.github/workflow/deploy.yml] --> ECR[Amazon ECR Public\nplasma-disruption-dashboard]
    ECR --> ECS[ECS Fargate Service\nplasma-disruption-service]
    ECS --> IP[Public IP]
    IP --> CF[Cloudflare DNS\nA Record update]
    USER[End User] --> CF
    CF --> ECS

    classDef cloud fill:#111827,stroke:#60A5FA,stroke-width:2px,color:#F8FAFC;
    class GH,ECR,ECS,IP,CF,USER cloud;
```

---

## 5) Runtime Components

- **Frontend**: Dashboard UI, controls, charts, tables
- **Backend**: Flask endpoints for generation, prediction, metadata
- **Model**: Bayesian NN with MC Dropout uncertainty
- **Artifacts**: model weights, scaler, feature schema
- **Container**: Dockerized Python app exposed on port `5000`
- **Cloud**: ECS Fargate deployment through GitHub Actions

---

## 6) Key File Mapping

- `app.py` → Flask API + model loading + inference logic
- `templates/index.html` → Dashboard markup
- `static/script.js` → Frontend behavior and API calls
- `static/style.css` → Dashboard styling
- `plasma_disruption_model.pt` → Trained model parameters
- `feature_scaler.pkl` → Input normalization
- `feature_columns.pkl` → Ordered feature schema
- `Dockerfile` → Multi-stage CPU-only container build
- `.github/workflow/deploy.yml` → CI/CD to ECR + ECS + DNS

---

If your markdown viewer does not render Mermaid, use GitHub web view for best display quality and contrast.
