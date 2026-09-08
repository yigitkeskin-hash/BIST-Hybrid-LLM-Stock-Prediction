import re
from typing import Dict, Any, Optional

def generate_stock_recommendation(
    ticker: str,
    company_name: str,
    current_price: float,
    predicted_price: float,
    expected_return_pct: float,
    rsi: float,
    sma_20: float,
    macd: float,
    sentiment: float,
    best_model: str,
    news_titles: list = None
) -> Dict[str, Any]:
    """
    Generates actionable investment recommendations (STRONG BUY / BUY / HOLD / SELL)
    and detailed rationale in Turkish and English using Microsoft Foundry Local / Phi-3.5 or
    deterministic financial quantitative rules.
    """
    rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
    sma_status = "Above SMA-20 (Bullish)" if current_price >= sma_20 else "Below SMA-20 (Corrective)"
    sentiment_status = "Bullish / Positive" if sentiment > 0.1 else "Bearish / Depressed" if sentiment < -0.1 else "Neutral"
    
    # Quantitative Decision Logic
    if expected_return_pct > 3.0 and rsi < 65 and sentiment >= -0.05:
        decision = "STRONG BUY"
        badge = "STRONG BUY"
        color = "#10b981" # emerald
        confidence = 88
    elif expected_return_pct > 0.8 and rsi < 70:
        decision = "BUY"
        badge = "BUY"
        color = "#22c55e" # green
        confidence = 74
    elif expected_return_pct < -3.0 or rsi > 75:
        decision = "SELL"
        badge = "SELL"
        color = "#ef4444" # red
        confidence = 80
    elif expected_return_pct < -0.8:
        decision = "REDUCE / WEAK"
        badge = "WEAK / REDUCE"
        color = "#f97316" # orange
        confidence = 68
    else:
        decision = "HOLD"
        badge = "HOLD"
        color = "#eab308" # yellow
        confidence = 70

    # Try calling Microsoft Foundry Local (phi-3.5-mini) if available
    llm_rationale = None
    try:
        from foundry_local_sdk import FoundryLocalManager
        if hasattr(FoundryLocalManager, "instance") and FoundryLocalManager.instance is not None:
            manager = FoundryLocalManager.instance
            model = manager.catalog.get_model("phi-3.5-mini")
            client = model.get_chat_client()
            
            prompt = f"""Equity: {ticker} ({company_name})
Current Price: {current_price:.2f} TRY
AI Target Price: {predicted_price:.2f} TRY (Expected Return: {expected_return_pct:+.2f}%)
14-Day RSI: {rsi:.1f} ({rsi_status})
20-Day Simple Moving Average (SMA-20): {sma_20:.2f} TRY ({sma_status})
MACD: {macd:.2f}
Market Sentiment Score: {sentiment:+.2f} ({sentiment_status})
Selected Best Neural Model: {best_model}

As an institutional Chief Investment Officer (CIO), provide a concise 2-3 sentence strategic rationale in English regarding this investment outlook. Ensure strict alignment with the stance: {decision}."""
            
            resp = client.complete_chat(messages=[
                {"role": "system", "content": "You are a senior Quantitative Portfolio Strategist specializing in equity markets. Deliver concise, professional executive investment rationales without disclaimers."},
                {"role": "user", "content": prompt}
            ])
            llm_rationale = resp.choices[0].message.content.strip()
    except Exception:
        pass

    # High-standard deterministic fallback rationale if LLM offline
    if not llm_rationale:
        if "BUY" in decision:
            llm_rationale = (
                f"The {best_model} neural architecture projects a {expected_return_pct:+.2f}% expected upside. "
                f"With the 14-day RSI at {rsi:.1f} reflecting sustainable momentum without severe overbought risks, "
                f"and price trading {sma_status.lower()} backed by a {sentiment:+.2f} sentiment index, tactical accumulation is supported."
            )
        elif "SELL" in decision or "REDUCE" in decision:
            llm_rationale = (
                f"The {best_model} model projects a {expected_return_pct:+.2f}% downside risk. "
                f"Given an RSI reading of {rsi:.1f} and technical indicators remaining {sma_status.lower()}, "
                f"short-term capital preservation and profit realization are advised."
            )
        else:
            llm_rationale = (
                f"The asset is undergoing balanced consolidation near current levels of {current_price:.2f} TRY ({expected_return_pct:+.2f}% projected change). "
                f"With RSI holding at {rsi:.1f} in neutral territory, maintaining existing allocations (HOLD) is warranted until directional confirmation emerges."
            )

    return {
        "ticker": ticker,
        "company_name": company_name,
        "decision": decision,
        "badge": badge,
        "color": color,
        "confidence": confidence,
        "current_price": round(current_price, 2),
        "predicted_price": round(predicted_price, 2),
        "expected_return_pct": round(expected_return_pct, 2),
        "best_model": best_model,
        "indicators": {
            "rsi": round(rsi, 1),
            "rsi_status": rsi_status,
            "sma_20": round(sma_20, 2),
            "sma_status": sma_status,
            "macd": round(macd, 2),
            "sentiment_score": round(sentiment, 2),
            "sentiment_status": sentiment_status
        },
        "rationale": llm_rationale
    }
