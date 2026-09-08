import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
from typing import Dict, Any, List

# Core BIST 100 benchmark candidate universe across different sectors
BIST_UNIVERSE = [
    "THYAO", "ASELS", "GARAN", "AKBNK", "ISCTR", "EREGL", 
    "BIMAS", "MGROS", "KCHOL", "SAHOL", "TUPRS", "FROTO", 
    "SISE", "ASTOR", "KONTR", "TCELL", "PGSUS", "TOASO", "ENKAI", "ALARK"
]

def find_best_optimal_bist_portfolio() -> Dict[str, Any]:
    """
    Scans the BIST universe, computes annualized returns and covariance matrix,
    and runs Markowitz Max Sharpe optimization. Filters out 0% or negligible weight
    assets so that ONLY the winning optimal basket (typically 3 to 6 top assets) is returned.
    """
    yf_symbols = [f"{t}.IS" for t in BIST_UNIVERSE]
    
    try:
        data = yf.download(yf_symbols, period="1y", progress=False)['Close']
        if isinstance(data, pd.Series):
            df_pivot = data.to_frame()
        else:
            df_pivot = data.dropna(axis=1, how='all')
            
        df_pivot.columns = [c.replace(".IS", "") for c in df_pivot.columns]
        df_pivot = df_pivot.dropna()
        valid_tickers = [t for t in BIST_UNIVERSE if t in df_pivot.columns]
    except Exception as e:
        return {"error": f"Veri çekme hatası: {str(e)}"}
        
    returns_df = df_pivot[valid_tickers].pct_change().dropna()
    cov_matrix = returns_df.cov() * 252
    exp_returns = (returns_df.mean() * 252).values
    
    risk_free_rate = 0.35  # TCMB referans faiz oranı (%35)
    
    def portfolio_performance(weights, exp_ret, cov_mat):
        port_ret = np.sum(exp_ret * weights)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights)))
        return port_ret, port_vol

    def negative_sharpe_ratio(weights, exp_ret, cov_mat, rf):
        p_ret, p_vol = portfolio_performance(weights, exp_ret, cov_mat)
        return -(p_ret - rf) / (p_vol + 1e-8)
        
    num_assets = len(valid_tickers)
    init_weights = np.ones(num_assets) / num_assets
    
    # 0% to 50% per stock bounds allows optimizer to select ONLY the best assets and drop the rest to 0%
    bounds = tuple((0.0, 0.50) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    
    opt_result = minimize(
        negative_sharpe_ratio,
        init_weights,
        args=(exp_returns, cov_matrix.values, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    opt_weights = opt_result.x
    opt_ret, opt_vol = portfolio_performance(opt_weights, exp_returns, cov_matrix.values)
    opt_sharpe = (opt_ret - risk_free_rate) / (opt_vol + 1e-8)
    
    # Company Name Map
    name_map = {
        "ASTOR": "Astor Enerji",
        "ASELS": "Aselsan",
        "GARAN": "Garanti BBVA",
        "AKBNK": "Akbank",
        "BIMAS": "BİM Mağazalar",
        "THYAO": "Türk Hava Yolları",
        "FROTO": "Ford Otosan",
        "TUPRS": "Tüpraş",
        "PGSUS": "Pegasus",
        "TOASO": "Tofaş",
        "MGROS": "Migros",
        "KCHOL": "Koç Holding",
        "SAHOL": "Sabancı Holding",
        "SISE": "Şişecam",
        "EREGL": "Erdemir",
        "TCELL": "Turkcell"
    }
    
    colors = ['#0f172a', '#0284c7', '#059669', '#d97706', '#7c3aed', '#dc2626', '#db2777', '#0891b2']
    
    # Filter ONLY assets with significant weight (>= 2%)
    significant_allocations = []
    for t, w in zip(valid_tickers, opt_weights):
        pct = float(w * 100)
        if pct >= 1.5:
            significant_allocations.append({
                "ticker": t,
                "name": name_map.get(t, t),
                "weight": round(pct, 1)
            })
            
    # Sort highest to lowest
    significant_allocations.sort(key=lambda x: x["weight"], reverse=True)
    
    # Normalize weights so they sum exactly to 100%
    total_sig = sum(a["weight"] for a in significant_allocations)
    if total_sig > 0:
        for idx, item in enumerate(significant_allocations):
            item["weight"] = round((item["weight"] / total_sig) * 100, 1)
            item["color"] = colors[idx % len(colors)]
            
    top_picks = ", ".join([f"{a['ticker']} ({a['weight']}%)" for a in significant_allocations])
    
    cio_memo = (
        f"The quantitative algorithm scanned the core BIST 100 universe and identified an optimal {len(significant_allocations)}-asset portfolio "
        f"maximizing the risk-adjusted Sharpe Ratio: {top_picks}. "
        f"This strategic allocation projects an expected annualized return of {opt_ret*100:.1f}% with a Sharpe ratio of {opt_sharpe:.2f}, "
        f"providing superior cross-sector diversification."
    )
    
    return {
        "annualized_return_pct": round(float(opt_ret * 100), 2),
        "annualized_volatility_pct": round(float(opt_vol * 100), 2),
        "sharpe_ratio": round(float(opt_sharpe), 2),
        "asset_count": len(significant_allocations),
        "allocations": significant_allocations,
        "strategy": f"AI-OPTIMIZED TOP {len(significant_allocations)} BIST BASKET",
        "cio_memo": cio_memo
    }
