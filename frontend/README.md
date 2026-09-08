# 💻 BIST 100 AI Terminal — Frontend

The web client for the **BIST 100 AI Terminal**, built with React 19, Vite, and Chart.js.

## 🚀 Features

- **Live Market Telemetry:** Real-time BIST 100 (XU100) benchmark index status, session tracking, and change rates.
- **Dynamic Stock Search:** Autocomplete and quick-select chips for major BIST equities (THYAO, ASELS, BIMAS, GARAN, FROTO, etc.).
- **Multi-Model Chart Visualizer:** Interactive time-series comparing actual prices against:
  - PyTorch **Residual Attention-GRU**
  - PyTorch **GRU**
  - PyTorch **LSTM**
- **AI Investment Stance:** Displays Microsoft Phi-3.5 offline SLM recommendations (`GÜÇLÜ AL`, `AL`, `TUT`, `SAT`), executive rationale, and confidence scores.
- **Markowitz Portfolio Optimizer UI:** Interactive asset allocation breakdown with Sharpe ratio analytics.

## 🛠️ Tech Stack

- **Framework:** [React 19](https://react.dev/) + [Vite](https://vitejs.dev/)
- **Charts:** [Chart.js](https://www.chartjs.org/) & [react-chartjs-2](https://react-chartjs-2.js.org/)
- **Icons:** [Lucide React](https://lucide.dev/)
- **Styling:** Custom responsive Glassmorphism design system

## ⚡ Development Setup

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The application will be served at `http://localhost:5173`. Make sure the FastAPI backend is running at `http://127.0.0.1:8000`.
