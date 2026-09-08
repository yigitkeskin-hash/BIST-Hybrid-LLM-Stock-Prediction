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
    rsi_status = "Aşırı Alım (Overbought)" if rsi > 70 else "Aşırı Satım (Oversold)" if rsi < 30 else "Dengeli (Neutral)"
    sma_status = "SMA-20 Üzerinde (Boğa Trendi)" if current_price >= sma_20 else "SMA-20 Altında (Düzeltme Trendi)"
    sentiment_status = "Pozitif / İyimser" if sentiment > 0.1 else "Negatif / Baskılı" if sentiment < -0.1 else "Nötr"
    
    # Quantitative Decision Logic
    if expected_return_pct > 3.0 and rsi < 65 and sentiment >= -0.05:
        decision = "GÜÇLÜ AL"
        badge = "STRONG BUY"
        color = "#10b981" # emerald
        confidence = 88
    elif expected_return_pct > 0.8 and rsi < 70:
        decision = "AL"
        badge = "BUY"
        color = "#22c55e" # green
        confidence = 74
    elif expected_return_pct < -3.0 or rsi > 75:
        decision = "SAT"
        badge = "SELL"
        color = "#ef4444" # red
        confidence = 80
    elif expected_return_pct < -0.8:
        decision = "ZAYIF / RİSKLİ"
        badge = "WEAK / REDUCE"
        color = "#f97316" # orange
        confidence = 68
    else:
        decision = "TUT"
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
            
            prompt = f"""Hisse: {ticker} ({company_name})
Güncel Fiyat: {current_price:.2f} TL
Yapay Zeka Hedef Fiyat: {predicted_price:.2f} TL (Beklenen Değişim: %{expected_return_pct:+.2f})
14 Günlük RSI: {rsi:.1f} ({rsi_status})
20 Günlük Hareketli Ortalama (SMA-20): {sma_20:.2f} TL ({sma_status})
MACD: {macd:.2f}
Piyasa Haber Duyarlılık Skoru: {sentiment:+.2f} ({sentiment_status})
En Başarılı Model: {best_model}

Lütfen bir Baş Yatırım Uzmanı (CIO) olarak bu hissenin alınıp alınmaması gerektiğine dair 2-3 cümlelik çok net, profesyonel bir Türkçe yatırım gerekçesi yaz. Kararın ({decision}) ile tam uyumlu olsun."""
            
            resp = client.complete_chat(messages=[
                {"role": "system", "content": "Sen Borsa İstanbul konusunda uzman, profesyonel bir Kantitatif Yatırım Danışmanısın. Gereksiz uyarı yapmadan doğrudan stratejik ve teknik analize odaklan."},
                {"role": "user", "content": prompt}
            ])
            llm_rationale = resp.choices[0].message.content.strip()
    except Exception:
        pass

    # High-standard deterministic fallback rationale if LLM offline
    if not llm_rationale:
        if "AL" in decision:
            llm_rationale = (
                f"{best_model} yapay zeka mimarisi, hisse için %{expected_return_pct:+.2f} getiri potansiyeli öngörüyor. "
                f"RSI göstergesinin {rsi:.1f} seviyesinde olması aşırı alım riski taşımadan sağlıklı bir momentum işaret ederken, "
                f"fiyatın {sma_status.lower()} seyretmesi ve haber duyarlılığının {sentiment:+.2f} olması pozitif pozisyonlanmayı destekliyor."
            )
        elif "SAT" in decision or "ZAYIF" in decision:
            llm_rationale = (
                f"{best_model} projeksiyonu %{expected_return_pct:+.2f} yönlü aşağı yönlü baskı tahmin etmektedir. "
                f"RSI değerinin {rsi:.1f} seviyesinde bulunması ve teknik görünümün {sma_status.lower()} kalması, kısa vadede kâr realizasyonu veya temkinli kalmayı gerektirmektedir."
            )
        else:
            llm_rationale = (
                f"Hisse mevcut {current_price:.2f} TL seviyesinde dengeli bir konsolidasyon sürecindedir (%{expected_return_pct:+.2f} beklenen değişim). "
                f"RSI'ın {rsi:.1f} nötr bölgesinde kalması sebebiyle yeni bir kırılım görülene kadar mevcut pozisyonların korunması (TUT) tavsiye edilmektedir."
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
