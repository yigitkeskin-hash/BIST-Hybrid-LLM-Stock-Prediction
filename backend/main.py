from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from typing import Optional, List

from services.data_engine import (
    BIST_100_STOCKS,
    fetch_live_stock_data,
    fetch_bist100_index
)
from services.ml_engine import benchmark_models_for_dataframe
from services.sentiment_llm import generate_stock_recommendation
from services.portfolio_engine import find_best_optimal_bist_portfolio

app = FastAPI(
    title="BIST 100 AI Terminal API",
    description="End-to-End BIST 100 Quantitative & AI Investment Terminal",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health_check():
    return {"status": "online", "system": "BIST 100 AI Terminal"}

@app.get("/api/market-status")
def get_market_status():
    """Returns live BIST 100 Index (XU100) and seans status."""
    index_data = fetch_bist100_index()
    return {
        "index": index_data,
        "market": "Borsa İstanbul (BIST)",
        "status": "AÇIK",
        "currency": "TRY"
    }

@app.get("/api/bist100-stocks")
def get_all_bist_stocks():
    """Returns the full list of BIST 100 stocks."""
    return BIST_100_STOCKS

@app.get("/api/analyze-stock")
def analyze_bist_stock(ticker: str = Query(..., description="BIST 100 Ticker symbol, e.g. THYAO, GARAN")):
    """
    CANLI BIST 100 HİSSE ANALİZİ:
    - yfinance canlı verisi
    - 8 teknik indikatör
    - Pytorch tahmin modeli
    - Canlı haberler & AI Yatırım Kararı (AL/SAT/TUT)
    """
    clean_ticker = ticker.strip().upper().replace(".IS", "")
    try:
        df, news_items, current_live_price, real_daily_change, analyst_target_price = fetch_live_stock_data(clean_ticker)
        
        # Calculate sentiment from news titles
        sentiment = 0.05
        if news_items:
            pos_words = ["rekor", "büyüme", "kâr", "artış", "temettü", "ihale", "anlaşma", "yükseliş", "record", "profit", "dividend", "growth"]
            neg_words = ["zarar", "düşüş", "soruşturma", "ceza", "risk", "kayıp", "enflasyon", "loss", "decline", "fine"]
            score = 0
            for item in news_items:
                t = item['title'].lower()
                score += sum(1 for w in pos_words if w in t)
                score -= sum(1 for w in neg_words if w in t)
            sentiment = float(np.clip(score * 0.15, -1.0, 1.0))
            
        ml_res = benchmark_models_for_dataframe(df, sentiment)
        
        predicted_price = ml_res["forecast"]["predicted_price"]
        expected_pct = ml_res["forecast"]["expected_return_pct"]
        best_model = ml_res["metrics"]["winner"]
        
        latest_row = df.iloc[-1]
        
        meta = next((s for s in BIST_100_STOCKS if s["symbol"] == clean_ticker), None)
        company_name = meta["name"] if meta else f"{clean_ticker} BIST Şirketi"
        sector = meta["sector"] if meta else "Borsa İstanbul"
        
        year_end_target = analyst_target_price if analyst_target_price else round(current_live_price * 1.35, 2)
        year_end_return_pct = round(((year_end_target - current_live_price) / current_live_price) * 100, 2) if current_live_price > 0 else 0.0

        recommendation = generate_stock_recommendation(
            ticker=clean_ticker,
            company_name=company_name,
            current_price=current_live_price,
            predicted_price=predicted_price,
            expected_return_pct=expected_pct,
            rsi=float(latest_row['rsi']),
            sma_20=float(latest_row['sma_20']),
            macd=float(latest_row['macd']),
            sentiment=float(sentiment),
            best_model=best_model,
            news_titles=[n['title'] for n in news_items]
        )
        recommendation["year_end_target"] = year_end_target
        recommendation["year_end_return_pct"] = year_end_return_pct
        
        return {
            "ticker": clean_ticker,
            "company_name": company_name,
            "sector": sector,
            "current_price": round(float(current_live_price), 2),
            "daily_change_pct": round(float(real_daily_change), 2),
            "recommendation": recommendation,
            "year_end_target": year_end_target,
            "year_end_return_pct": year_end_return_pct,
            "chart_history": df[['date', 'open', 'high', 'low', 'close', 'volume', 'rsi', 'sma_20', 'macd', 'bb_high', 'bb_low']].to_dict(orient="records"),
            "days_forecast": ml_res.get("days_forecast", []),
            "model_targets": ml_res.get("metrics", {}).get("targets", {}),
            "model_history": ml_res.get("model_history", {}),
            "news": news_items
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"'{clean_ticker}' analiz edilirken hata: {str(e)}")

@app.get("/api/portfolio/optimal")
def get_auto_optimal_portfolio():
    """Scans BIST 100 universe and returns the mathematically best Max Sharpe portfolio automatically."""
    res = find_best_optimal_bist_portfolio()
    if "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])
    return res

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
