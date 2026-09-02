# 📈 BIST Stock Price Prediction: Hybrid PyTorch, Residual Attention & Local LLM Portfolio Pipeline

An end-to-end quantitative research and portfolio optimization pipeline for predicting Turkish stock market (**Borsa Istanbul - BIST**) equities by combining **multi-factor technical indicators**, **deep learning architectures with attention mechanisms**, and **offline Local LLM-powered sentiment analysis and asset allocation**.

Unlike traditional forecasting approaches that rely solely on historical prices, this project integrates **live financial news sentiment** analyzed completely offline using a **local Phi-3.5-mini language model via Microsoft Foundry Local**, preserving full data privacy without third-party cloud APIs.

The forecasting engine benchmarks three deep learning architectures in PyTorch: **LSTM**, **GRU**, and an advanced **Residual Attention-GRU** (which fuses recent price momentum with temporal attention weights). Model forecasts are subsequently channeled into a **Markowitz Mean-Variance Optimizer (Max Sharpe Ratio)** with an AI Chief Investment Officer summary.

---

# 🚀 Features

- **Automated Quantitative Pipeline:** End-to-end data acquisition, feature engineering, neural training, and portfolio allocation.
- **Market Data Ingestion:** 1-year historical BIST daily price collection via **yfinance**.
- **Multi-Factor Feature Suite (8 Dimensions):**
  - Momentum: **RSI (14-day)**, **MACD & Signal Line (12, 26, 9)**
  - Trend & Volatility: **SMA-20**, **Bollinger Bands (Upper & Lower)**
  - Sentiment: Normalized news sentiment score `[-1.0, +1.0]`
- **Privacy-Preserving News Sentiment:** Real financial news fetched via Yahoo Finance and scored offline using **Microsoft Foundry Local + Phi-3.5-mini**.
- **Data-Leakage Free Preprocessing:** `MinMaxScaler` fitted strictly on training splits; independent target scaler for seamless inverse transform.
- **Deep Learning Benchmark:** Comparative evaluation of **LSTM**, **GRU**, and **Residual Attention-GRU**.
- **Local LLM Investment Advisor:** Automated generation of multi-factor strategic recommendations (`STRONG BUY`, `BUY`, `HOLD`, `SELL`) with concise 2–3 sentence rationales.
- **Markowitz Modern Portfolio Theory (MPT):** Quadratic optimization maximizing the **Sharpe Ratio** under institutional diversification constraints (1% minimum floor, 60% max cap).

---

# 📊 Pipeline Architecture

```
[ BIST Market Data (yfinance) ]       [ Yahoo Finance News ]
               │                                │
               ▼                                ▼
[ Multi-Indicator Engineering ]       [ Microsoft Foundry Local ]
 (RSI, SMA-20, MACD, BBands)             (Phi-3.5-mini LLM)
               │                                │
               └───────────────┬────────────────┘
                               ▼
               [ SQLite Database: bist_hybrid.db ]
                               │
                               ▼
               [ Leakage-Free Sliding Window (20, 8) ]
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
        [ LSTM ]            [ GRU ]      [ Residual Attn-GRU ]
            └──────────────────┬──────────────────┘
                               ▼
               [ Model Evaluation & Forecasts (RMSE) ]
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
[ Local LLM Asset Advisor ]           [ Markowitz Portfolio Optimizer ]
 (Actionable Recommendations)          (Max Sharpe Ratio & Capital Allocation)
```

---

## Phase 1 — Database Initialization & Storage
Creates a local SQLite database (`bist_hybrid.db`) with structured tables for:
- `bist_prices`: Open, High, Low, Close, Volume, RSI, SMA-20, MACD, MACD Signal, Bollinger High, Bollinger Low.
- `news_sentiment`: Ticker, Date, Headline, Content, Sentiment Score.

---

## Phase 2 — Market Data & Extended Indicator Generation
Downloads historical market data for 5 BIST blue-chip tickers:
- **FROTO** (Ford Otosan)
- **ASELS** (Aselsan)
- **TUPRS** (Tüpraş)
- **THYAO** (Turkish Airlines)
- **SAHOL** (Sabancı Holding)

Engineers 7 numerical indicators combined with sentiment:
- **SMA-20** (20-Day Simple Moving Average)
- **Bollinger Bands** ($\mu \pm 2\sigma$)
- **RSI** (14-Day Relative Strength Index)
- **MACD & Signal** ($\text{EMA}_{12} - \text{EMA}_{26}$)

---

## Phase 3 — Local LLM Financial Sentiment Analysis
Financial news headlines and summaries are analyzed **completely offline** using:
- **Microsoft Foundry Local**
- **Phi-3.5-mini**

A Few-Shot System Prompt maps articles to normalized scores:
- **-1.0** $\rightarrow$ Strongly Bearish
- **0.0** $\rightarrow$ Neutral / Balanced
- **+1.0** $\rightarrow$ Strongly Bullish

---

## Phase 4 — Leakage-Free Data Preprocessing
- Unified 8-feature representation: `[Close, RSI, SMA-20, MACD, Signal, BB High, BB Low, Sentiment]`.
- Strict **80/20 chronological train-test split**.
- `MinMaxScaler(feature_range=(-1, 1))` fitted **strictly on train splits** (`transform` applied to test splits) preventing future price leakage.
- Independent `price_scaler` for direct and robust inverse transformation.
- Sequence generation using a **20-day sliding window**:
  $$\text{Input Shape} = (N, 20, 8)$$

---

## Phase 5 — Deep Learning Benchmarking

Three neural network architectures implemented in PyTorch:

1. **LSTM:** 2 layers, 32 hidden units, linear head.
2. **GRU:** 2 layers, 32 hidden units, linear head.
3. **Residual Attention-GRU:** Fuses the last hidden step ($h_T$) with a softmax temporal attention context vector:
   $$\text{Combined} = [h_T \,\|\, \sum_{t=1}^T \alpha_t h_t] \longrightarrow \text{MLP}$$
   Preserves critical recency momentum while incorporating contextual shocks from the 20-day lookback window.

**Training Configuration:**
- Optimizer: **Adam** (`lr=0.01`)
- Loss Function: **MSE Loss**
- Epochs: **60**
- Evaluation Metric: **Root Mean Square Error (RMSE)**

---

## Phase 6 — Local LLM Multi-Factor Asset Advisor
Synthesizes neural price projections, RSI momentum status, SMA trend position, and news sentiment into executive trading signals:
- **DECISION:** `[STRONG BUY / BUY / HOLD / SELL / STRONG SELL]`
- **TARGET RETURN:** Projected percentage return
- **RATIONALE:** 2–3 concise professional sentences explaining the decision.

---

## Phase 7 — Markowitz Modern Portfolio Theory (MPT) Optimization
Solves for the **Maximum Sharpe Ratio Portfolio**:
- Inputs: Model-projected expected returns $\mu$ and empirical covariance matrix $\Sigma$.
- Risk-Free Rate Reference: **35.0%** (Turkish Central Bank repo benchmark).
- Constraints: Full capital allocation ($\sum w_i = 100\%$), minimum floor ($w_i \ge 1\%$), and max asset concentration cap ($w_i \le 60\%$).
- Generates a **Capital Allocation Pie Chart** and an **Executive Allocation Memo** via Local LLM.

---

# 🛠 Technology Stack

| Category | Technology |
|----------|------------|
| Language | Python 3 |
| Deep Learning | PyTorch |
| Local LLM | Microsoft Foundry Local |
| Language Model | Phi-3.5-mini |
| Market Data & News | yfinance (Yahoo Finance) |
| Numerical & Data | pandas, NumPy, SciPy (SLSQP Optimizer) |
| Machine Learning | scikit-learn (MinMaxScaler) |
| Database | SQLite3 |
| Visualization | Matplotlib |

---

# 📈 Experimental & Portfolio Results

### Deep Learning Model Benchmark (RMSE):
| Ticker | Company | LSTM RMSE | GRU RMSE | Residual Attn-GRU | Best Model |
|--------|---------|----------:|---------:|------------------:|:----------:|
| **FROTO** | Ford Otosan | 5.55 TRY | 3.56 TRY | **3.28 TRY** | **Attention-GRU** |
| **ASELS** | Aselsan | 17.43 TRY | **11.89 TRY** | 15.58 TRY | **GRU** |
| **TUPRS** | Tüpraş | 59.86 TRY | **55.68 TRY** | 59.59 TRY | **GRU** |
| **THYAO** | Turkish Airlines | 16.33 TRY | **8.75 TRY** | 14.69 TRY | **GRU** |
| **SAHOL** | Sabancı Holding | 1.86 TRY | **1.63 TRY** | 2.19 TRY | **GRU** |

### Optimal Capital Allocation (Markowitz Max Sharpe):
| Asset | Company | Optimal Weight | Model Stance & Outlook |
|:-----:|:--------|:--------------:|:-----------------------|
| **FROTO** | Ford Otosan | **60.00%** | STRONG BUY (+5.99% forecast, oversold RSI) |
| **THYAO** | Turkish Airlines | **26.25%** | HOLD (+0.63% positive momentum) |
| **SAHOL** | Sabancı Holding | **11.75%** | HOLD (+0.90% positive momentum) |
| **ASELS** | Aselsan | **1.00%** | HOLD (diversification floor) |
| **TUPRS** | Tüpraş | **1.00%** | SELL (diversification floor) |

---

# 📌 Reproducibility

To ensure deterministic and reproducible experimental runs, fixed random seeds (`42`) are established for:
- `torch.manual_seed(42)`
- `np.random.seed(42)`
- Python `random.seed(42)`

---

# 📚 Future Improvements

- Transformer-based architectures (PatchTST, Temporal Fusion Transformer)
- Real-time event-driven news ingestion via WebSockets / KAP API
- Multi-step ahead autoregressive forecasting horizons
- Hyperparameter tuning automation using Optuna
