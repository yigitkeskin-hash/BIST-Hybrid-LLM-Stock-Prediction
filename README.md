# 📈 BIST 100 AI Terminal: Hybrid PyTorch, Residual Attention & Microsoft Foundry Local SLM

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c?logo=pytorch&logoColor=white)
![Microsoft Foundry Local](https://img.shields.io/badge/Microsoft%20Foundry%20Local-Phi--3.5--mini-0078d4?logo=microsoft&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)
![React + Vite](https://img.shields.io/badge/React%2019-Vite-61dafb?logo=react&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**An institutional-grade, end-to-end quantitative intelligence platform for Borsa Istanbul (BIST 100) equities.**  
Fusing **PyTorch Deep Learning (Residual Attention-GRU)**, **Microsoft Phi-3.5 On-Device Financial Sentiment Analysis**, and **Markowitz Modern Portfolio Theory (Max Sharpe Ratio)** into a real-time reactive Web Terminal.

[Live Terminal Overview](#-terminal-features) • [Architecture](#-pipeline-architecture) • [Microsoft AI Synergy](#-why-microsoft-foundry-local--phi-35) • [Model Benchmarks](#-deep-learning-benchmark-results) • [Quickstart](#-quickstart-guide)

</div>

---

## 🎯 Executive Summary

Most algorithmic trading and equity research platforms rely either solely on historical price trends (ignoring market sentiment) or outsource text analysis to cloud LLMs (risking confidential financial data leakage and incurring prohibitive cloud latencies).

**BIST 100 AI Terminal** solves this by delivering an **offline-capable, privacy-preserving hybrid quant engine**:
1. **Edge SLM Sentiment:** Real-time news sentiment is processed locally via **Microsoft Foundry Local** running **Phi-3.5-mini**, keeping analytical intelligence strictly on-premise.
2. **Deep Sequence Forecasting:** Benchmarks **LSTM**, **GRU**, and a custom **Residual Attention-GRU** in PyTorch over an 8-factor feature tensor with zero lookahead bias.
3. **Automated Asset Allocation:** Model price forecasts feed directly into a **Markowitz Mean-Variance Optimizer (SLSQP)** to generate maximum Sharpe ratio allocations under institutional risk bounds.
4. **Interactive Bloomberg-Style UI:** Built with FastAPI, React, Tailwind-grade custom CSS, and Chart.js for real-time equity diagnostics, multi-model curve toggling, and portfolio simulation.

---

## 🚀 Key Innovations & System Features

### 1. 🛡️ Privacy-First Sentiment via Microsoft Foundry Local
- **Model:** `phi-3.5-mini-instruct` running offline on local hardware through Microsoft Foundry Local SDK.
- **Data Privacy:** Financial queries, proprietary ticker watchlists, and live news articles remain completely offline—zero exposure to third-party cloud APIs.
- **Zero-Latency Fallback:** Robust deterministic heuristic fallback ensures 100% uptime even on environments without local NPU/GPU acceleration.

### 2. 🧠 Hybrid Deep Learning Suite (PyTorch)
- **Multi-Factor Feature Space (8 Dimensions):**
  - **Momentum:** RSI (14-day), MACD & Signal Line (12, 26, 9)
  - **Trend & Volatility:** SMA-20, Bollinger Bands (Upper & Lower, $2\sigma$)
  - **Context:** Local LLM Sentiment Score $[-1.0, +1.0]$ + Closing Price
- **Leakage-Free Sliding Window (20-Day Lookback):** `MinMaxScaler` fit strictly on chronological train splits ($80\%$) and applied to test splits ($20\%$).
- **Residual Attention-GRU Architecture:**
  $$\text{Context Vector } c = \sum_{t=1}^{T} \alpha_t h_t, \quad \alpha = \text{softmax}(W_a h + b_a)$$
  $$\text{Fused Output } = \text{MLP}([h_T \,\|\, c])$$
  Fuses the most recent timestep $h_T$ (capturing immediate price momentum) with an attention-weighted historical context vector $c$ (capturing macroeconomic shocks).

### 3. ⚖️ Institutional Markowitz Portfolio Optimizer
- Scans candidate blue-chip equities across multiple BIST sectors (Aviation, Defense, Banking, Energy, Retail, Automotive).
- Numerically maximizes the **Sharpe Ratio** against the Turkish Central Bank (TCMB) repo benchmark rate ($35.0\%$).
- Automatically concentrates capital into the mathematically winning subset ($3-6$ assets), dropping low-Sharpe assets to $0\%$ while enforcing safety caps ($w_i \le 50\%$).

### 4. 💻 Full-Stack Interactive Web Terminal
- **Ticker Autocomplete:** Instant search across the entire BIST 100 universe (THYAO, ASELS, GARAN, BIMAS, FROTO, etc.).
- **Multi-Model Curve Visualizer:** Real-time toggling between actual price, Residual Attention-GRU, LSTM, and GRU projections.
- **Executive Rationale:** Local Phi-3.5 generated investment recommendations (`GÜÇLÜ AL / AL / TUT / SAT`) with plain-language reasoning and risk metrics.

---

## 📊 Pipeline Architecture

```
                       ┌───────────────────────────────┐
                       │   Borsa İstanbul (BIST 100)   │
                       └──────────────┬────────────────┘
                                      │
               ┌──────────────────────┴──────────────────────┐
               ▼                                             ▼
   [ Historical & Live Prices ]                   [ Live Financial News ]
         (yfinance API)                               (Yahoo Finance)
               │                                             │
               ▼                                             ▼
   [ Technical Feature Engineering ]             [ Microsoft Foundry Local ]
   • RSI (14)   • MACD & Signal                  • Phi-3.5-mini Offline SLM
   • SMA-20     • Bollinger Bands                • Normalized Sentiment [-1, 1]
               │                                             │
               └──────────────────────┬──────────────────────┘
                                      ▼
                        [ SQLite / In-Memory Store ]
                                      │
                                      ▼
                      [ Leakage-Free Preprocessing ]
                      • 80/20 Chronological Split
                      • Train-only Scaler Transform
                      • 20-Day Lookback Window (N, 20, 8)
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
     [ LSTM ]                      [ GRU ]              [ Residual Attn-GRU ]
   2-layer (32h)                2-layer (32h)              Dynamic Temporal Head
         └────────────────────────────┬────────────────────────────┘
                                      ▼
                       [ Model Evaluation & Forecast ]
                                      │
         ┌────────────────────────────┴────────────────────────────┐
         ▼                                                         ▼
[ Local LLM Investment Memo ]                           [ Markowitz MPT Optimizer ]
• Actionable Signals (BUY/SELL)                         • Maximize Sharpe Ratio (rf=35%)
• Key Technical Triggers                                • Institutional Diversification
         └────────────────────────────┬────────────────────────────┘
                                      ▼
                      [ FastAPI Asynchronous Backend ]
                                      │ (REST JSON API)
                                      ▼
                   [ React 19 + Vite Financial Terminal ]
                   • Interactive Multi-Model Charting
                   • Asset Allocation Doughnut & Breakdown
```

---

## 💡 Why Microsoft Foundry Local & Phi-3.5?

| Enterprise Dimension | Cloud LLM APIs (GPT-4, Claude) | Microsoft Foundry Local + Phi-3.5 |
| :--- | :--- | :--- |
| **Data Privacy** | Sensitive portfolio & ticker queries leave perimeter | **100% on-device execution; zero data leaves host** |
| **Operational Cost** | Pay-per-token API fees scale with market volume | **$0 inference cost; unlimited local processing** |
| **Inference Latency** | Network round-trip jitter ($500\text{ms} - 2000\text{ms}$) | **Sub-100ms on local hardware acceleration** |
| **Regulatory Compliance**| Hard to comply with banking/capital market laws (SPK, GDPR) | **Strictly compliant with on-premise governance** |

---

## 📈 Deep Learning Benchmark Results

Evaluated across historical daily test splits for major blue-chip equities using **Root Mean Square Error (RMSE)**:

| Ticker | Company Name | Sector | LSTM RMSE | GRU RMSE | Residual Attn-GRU | Selected Model |
|:------:|:-------------|:-------|----------:|---------:|------------------:|:--------------:|
| **FROTO** | Ford Otosan | Otomotiv | 5.55 TRY | 3.56 TRY | **3.28 TRY** | 🏆 **Attn-GRU** |
| **ASELS** | Aselsan | Savunma | 17.43 TRY | **11.89 TRY** | 15.58 TRY | 🏆 **GRU** |
| **TUPRS** | Tüpraş | Enerji | 59.86 TRY | **55.68 TRY** | 59.59 TRY | 🏆 **GRU** |
| **THYAO** | Türk Hava Yolları | Havacılık | 16.33 TRY | **8.75 TRY** | 14.69 TRY | 🏆 **GRU** |
| **SAHOL** | Sabancı Holding | Holding | 1.86 TRY | **1.63 TRY** | 2.19 TRY | 🏆 **GRU** |

> **Key Observation:** For equities with pronounced local trend shifts and volatility spikes (e.g. `FROTO`), the **Residual Attention-GRU** achieves superior performance by dynamically adjusting weights across historical days rather than treating all sequential hidden states uniformly.

---

## 🛠️ Technology Stack

- **Deep Learning & Modeling:** PyTorch, scikit-learn, NumPy, SciPy (`scipy.optimize.minimize` SLSQP)
- **Local Language Model:** Microsoft Foundry Local, Microsoft Phi-3.5-mini
- **Financial Data Ingestion:** yfinance, pandas, SQLite3
- **Backend API:** FastAPI (ASGI), Uvicorn, Pydantic
- **Frontend Dashboard:** React 19, Vite, Chart.js, react-chartjs-2, Lucide React, Modern Glassmorphism CSS

---

## ⚡ Quickstart Guide

### Prerequisites
- Python 3.10 or higher
- Node.js 18+ and npm

### 1. Clone Repository
```bash
git clone https://github.com/your-username/bist-ai-terminal.git
cd bist-ai-terminal
```

### 2. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```
> The API server will start at `http://127.0.0.1:8000` (API docs at `/docs`).

### 3. Frontend Setup
```bash
cd ../frontend
npm install
npm run dev
```
> Open your browser at `http://localhost:5173` to access the interactive terminal.

---

## 📌 Reproducibility & Research Integrity

All random seeds are explicitly fixed across frameworks to ensure reproducible benchmarking:
```python
import torch, numpy as np, random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
