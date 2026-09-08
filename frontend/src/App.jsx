import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import {
  TrendingUp,
  TrendingDown,
  Search,
  PieChart as PieIcon,
  Activity,
  RefreshCw,
  Zap,
  AlertTriangle,
  Newspaper,
  Award,
  ShieldCheck,
  Sparkles,
  Calendar,
  Compass
} from 'lucide-react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
} from 'chart.js';
import { Line, Pie } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
);

const API_BASE = "http://127.0.0.1:8000/api";

export default function App() {
  const [activeTab, setActiveTab] = useState('analyze');
  const [marketStatus, setMarketStatus] = useState(null);
  const [bist100List, setBist100List] = useState([]);
  
  // Stock Search State
  const [searchQuery, setSearchQuery] = useState('BIMAS');
  const [stockData, setStockData] = useState(null);
  const [loadingStock, setLoadingStock] = useState(false);
  const [stockError, setStockError] = useState(null);
  const [showDropdown, setShowDropdown] = useState(false);
  const dropdownRef = useRef(null);

  // Model Visibility Toggles for Chart
  const [showAttnGRU, setShowAttnGRU] = useState(true);
  const [showLSTM, setShowLSTM] = useState(true);
  const [showGRU, setShowGRU] = useState(true);
  const [showBacktest, setShowBacktest] = useState(true); // Toggle historical model curve
  const [chartRange, setChartRange] = useState('30'); // '15', '30', '90', 'all'

  // Auto Optimal Portfolio State
  const [portfolioData, setPortfolioData] = useState(null);
  const [loadingPortfolio, setLoadingPortfolio] = useState(false);

  const popularChips = ['THYAO', 'GARAN', 'ASELS', 'EREGL', 'BIMAS', 'KCHOL', 'SISE', 'ASTOR', 'TUPRS', 'FROTO'];

  const fetchMarketStatus = async () => {
    try {
      const res = await axios.get(`${API_BASE}/market-status`);
      setMarketStatus(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  const fetchBist100List = async () => {
    try {
      const res = await axios.get(`${API_BASE}/bist100-stocks`);
      setBist100List(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  const handleAnalyzeStock = async (symbol) => {
    const target = symbol || searchQuery;
    if (!target) return;
    setShowDropdown(false);
    setLoadingStock(true);
    setStockError(null);
    try {
      const res = await axios.get(`${API_BASE}/analyze-stock?ticker=${target.trim()}`);
      setStockData(res.data);
      setSearchQuery(target.toUpperCase());
    } catch (err) {
      setStockError(err.response?.data?.detail || "Hisse verisi çekilemedi.");
      setStockData(null);
    } finally {
      setLoadingStock(false);
    }
  };

  const fetchOptimalPortfolio = async () => {
    setLoadingPortfolio(true);
    try {
      const res = await axios.get(`${API_BASE}/portfolio/optimal`);
      setPortfolioData(res.data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingPortfolio(false);
    }
  };

  useEffect(() => {
    fetchMarketStatus();
    fetchBist100List();
    handleAnalyzeStock('BIMAS');
    fetchOptimalPortfolio();

    const handleClickOutside = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setShowDropdown(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const filteredStocks = bist100List.filter(
    s => s.symbol.toLowerCase().includes(searchQuery.toLowerCase()) ||
         s.name.toLowerCase().includes(searchQuery.toLowerCase())
  ).slice(0, 8);

  // Future 3-Day Projection Chart Setup with Flexible Past History & Model Backtest Curve
  const getInteractiveChartData = () => {
    if (!stockData || !stockData.chart_history) return null;
    
    // Slice past days according to selected timeframe (default 30 days)
    const count = chartRange === '15' ? 15 : chartRange === '30' ? 30 : chartRange === '90' ? 90 : stockData.chart_history.length;
    const history = stockData.chart_history.slice(-count);
    const pastDates = history.map(d => d.date.slice(5));
    const pastPrices = history.map(d => d.close);
    const lastPrice = pastPrices[pastPrices.length - 1];

    // Future labels: exactly the 3 days ahead
    const allLabels = [...pastDates, '+1G (Yarın)', '+2G', '+3G (Hedef)'];

    // 1. Actual Historical Price Curve (Ends at today)
    const actualSeries = [...pastPrices, null, null, null];

    // Helper for future series: nulls before today, today's price at the anchor, then days_forecast
    const makeFutureSeries = (modelKey) => {
      if (!stockData.days_forecast || stockData.days_forecast.length < 3) return [];
      const prefix = new Array(pastPrices.length - 1).fill(null);
      return [
        ...prefix,
        lastPrice, // connects seamlessly from today
        stockData.days_forecast[0][modelKey],
        stockData.days_forecast[1][modelKey],
        stockData.days_forecast[2][modelKey]
      ];
    };

    // Helper for past model predictions (Backtest curve on past days)
    const makeHistoricalSeries = (modelKey) => {
      if (!stockData.model_history) return [];
      const pastSeries = history.map(d => {
        const mhItem = stockData.model_history[d.date];
        return mhItem ? mhItem[modelKey] : null;
      });
      // Append null for future days so length matches allLabels
      return [...pastSeries, null, null, null];
    };

    const datasets = [
      {
        label: `${stockData.ticker} Gerçek Fiyat Geçmişi`,
        data: actualSeries,
        borderColor: '#0f172a',
        backgroundColor: 'rgba(15, 23, 42, 0.03)',
        fill: true,
        borderWidth: 2.5,
        pointRadius: 2,
        pointHoverRadius: 5
      }
    ];

    // Historical Backtest Models (Showing past GRU / Attention-GRU trajectory)
    if (showBacktest && showGRU) {
      datasets.push({
        label: 'GRU Geçmiş Tahmin Eğrisi (Backtest)',
        data: makeHistoricalSeries('GRU'),
        borderColor: 'rgba(217, 119, 6, 0.75)',
        borderDash: [3, 2],
        borderWidth: 1.8,
        pointRadius: 1.5,
        pointBackgroundColor: '#d97706'
      });
    }

    if (showBacktest && showAttnGRU) {
      datasets.push({
        label: 'Attention-GRU Geçmiş Tahmin Eğrisi',
        data: makeHistoricalSeries('Attention-GRU'),
        borderColor: 'rgba(5, 150, 105, 0.75)',
        borderDash: [3, 2],
        borderWidth: 1.8,
        pointRadius: 1.5,
        pointBackgroundColor: '#059669'
      });
    }

    // Future Projections (+1G, +2G, +3G)
    if (showAttnGRU) {
      datasets.push({
        label: 'Attention-GRU Gelecek Projeksiyonu (Öneri)',
        data: makeFutureSeries('Attention-GRU'),
        borderColor: '#059669',
        borderDash: [5, 4],
        borderWidth: 2.5,
        pointRadius: 5,
        pointBackgroundColor: '#059669',
        pointHoverRadius: 7
      });
    }

    if (showLSTM) {
      datasets.push({
        label: 'LSTM Gelecek Projeksiyonu',
        data: makeFutureSeries('LSTM'),
        borderColor: '#0284c7',
        borderDash: [3, 3],
        borderWidth: 1.8,
        pointRadius: 4,
        pointBackgroundColor: '#0284c7'
      });
    }

    if (showGRU) {
      datasets.push({
        label: 'GRU Gelecek Projeksiyonu',
        data: makeFutureSeries('GRU'),
        borderColor: '#d97706',
        borderDash: [2, 2],
        borderWidth: 2,
        pointRadius: 4,
        pointBackgroundColor: '#d97706'
      });
    }

    return { labels: allLabels, datasets };
  };

  const getPortfolioPieData = () => {
    if (!portfolioData || !portfolioData.allocations) return null;
    return {
      labels: portfolioData.allocations.map(a => `${a.ticker} (${a.name})`),
      datasets: [
        {
          data: portfolioData.allocations.map(a => a.weight),
          backgroundColor: portfolioData.allocations.map(a => a.color),
          borderColor: '#ffffff',
          borderWidth: 2
        }
      ]
    };
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="terminal-header">
        <div className="logo-area">
          <div className="logo-icon">
            <Activity size={22} />
          </div>
          <div>
            <h1 className="terminal-title">BIST 100 AI TERMINAL</h1>
            <p className="terminal-subtitle">Canlı BIST 100 Hisse Analizi & Otomatik AI Portföy Motoru</p>
          </div>
        </div>

        <div className="header-actions">
          {marketStatus && (
            <div className="bist-index-badge">
              <span className="bist-label">BIST 100 (XU100)</span>
              <span className="bist-val">{marketStatus.index.price.toLocaleString('tr-TR')} ₺</span>
              <span className={`change-badge ${marketStatus.index.is_positive ? 'positive' : 'negative'}`}>
                {marketStatus.index.is_positive ? <TrendingUp size={11} /> : <TrendingDown size={11} />}
                %{Math.abs(marketStatus.index.change_pct)}
              </span>
            </div>
          )}

          <div className="market-session-tag">
            <span className="pulsing-dot"></span>
            SEANS: AÇIK
          </div>

          <button
            onClick={() => {
              fetchMarketStatus();
              if (activeTab === 'analyze') handleAnalyzeStock();
              if (activeTab === 'portfolio') fetchOptimalPortfolio();
            }}
            className="icon-btn"
            title="Piyasayı Yenile"
          >
            <RefreshCw size={14} />
          </button>
        </div>
      </header>

      {/* Main Tabs */}
      <div className="nav-tabs">
        <button
          className={`tab-btn ${activeTab === 'analyze' ? 'active' : ''}`}
          onClick={() => setActiveTab('analyze')}
        >
          <Search size={15} /> BIST 100 Hisse Sor & Gelecek 3 Gün Tahmini
        </button>
        <button
          className={`tab-btn ${activeTab === 'portfolio' ? 'active' : ''}`}
          onClick={() => setActiveTab('portfolio')}
        >
          <Sparkles size={15} /> BIST 100 En Optimal Portföy (Yapay Zeka Önerisi)
        </button>
      </div>

      {/* TAB 1: BIST 100 STOCK LIVE ANALYZER & AI DECISION */}
      {activeTab === 'analyze' && (
        <div className="tab-content">
          <div className="glass-panel">
            <h2 className="panel-title">
              <Search size={16} />
              BIST 100 Hisse Arama & Çoklu Model Tahmin Raporu
            </h2>

            {/* Search Input with Autocomplete */}
            <div className="search-container" ref={dropdownRef}>
              <div className="search-input-wrapper">
                <Search className="search-icon-pos" size={16} />
                <input
                  type="text"
                  className="search-input"
                  placeholder="BIST 100 hisse kodu veya şirket adı yazın (örn: BIMAS, THYAO, GARAN, ASELS, ASTOR)..."
                  value={searchQuery}
                  onChange={(e) => {
                    setSearchQuery(e.target.value);
                    setShowDropdown(true);
                  }}
                  onFocus={() => setShowDropdown(true)}
                  onKeyDown={(e) => e.key === 'Enter' && handleAnalyzeStock()}
                />

                {showDropdown && filteredStocks.length > 0 && (
                  <div className="autocomplete-dropdown">
                    {filteredStocks.map((stock) => (
                      <div
                        key={stock.symbol}
                        className="autocomplete-item"
                        onClick={() => handleAnalyzeStock(stock.symbol)}
                      >
                        <div className="auto-left">
                          <span className="auto-sym">{stock.symbol}</span>
                          <span className="auto-name">{stock.name}</span>
                        </div>
                        <span className="auto-sec">{stock.sector}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <button
                className="search-btn"
                onClick={() => handleAnalyzeStock()}
                disabled={loadingStock}
              >
                {loadingStock ? <RefreshCw className="animate-spin" size={14} /> : <Zap size={14} />}
                {loadingStock ? 'Yapay Zeka İnceliyor...' : 'Fiyat & Tahmin Sor'}
              </button>
            </div>

            {/* Popular Chips */}
            <div className="popular-chips">
              <span className="chip-label">Hızlı Seçim:</span>
              {popularChips.map((chip) => (
                <button
                  key={chip}
                  className={`chip-btn ${searchQuery === chip ? 'active' : ''}`}
                  onClick={() => handleAnalyzeStock(chip)}
                >
                  {chip}
                </button>
              ))}
            </div>

            {stockError && (
              <div style={{ padding: 14, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#dc2626', marginBottom: 20, fontSize: 13 }}>
                <AlertTriangle size={16} style={{ display: 'inline', marginRight: 8 }} />
                {stockError}
              </div>
            )}

            {/* AI Decision Box */}
            {stockData && (
              <div>
                <div className="ai-decision-card">
                  <div className="decision-badge-col">
                    <div style={{ fontSize: 11, color: '#64748b', fontWeight: 700, letterSpacing: 0.5 }}>YAPAY ZEKA TAVSİYESİ</div>
                    <div
                      className="decision-glow-title"
                      style={{ color: stockData.recommendation.color }}
                    >
                      {stockData.recommendation.decision}
                    </div>
                    <div style={{ fontSize: 11, color: '#94a3b8', letterSpacing: 0.5 }}>{stockData.recommendation.badge}</div>

                    <div className="confidence-bar-wrap">
                      <div className="conf-label">
                        <span>Model Güven Oranı</span>
                        <span>%{stockData.recommendation.confidence}</span>
                      </div>
                      <div className="conf-track">
                        <div
                          className="conf-fill"
                          style={{
                            width: `${stockData.recommendation.confidence}%`,
                            backgroundColor: stockData.recommendation.color
                          }}
                        ></div>
                      </div>
                    </div>
                  </div>

                  <div className="decision-details-col">
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
                        <div>
                          <h3 style={{ fontSize: 20, fontWeight: 800, color: '#0f172a' }}>
                            {stockData.ticker} · <span style={{ color: '#64748b', fontWeight: 500 }}>{stockData.company_name}</span>
                          </h3>
                          <span style={{ fontSize: 12, color: '#64748b' }}>
                            Sektör: <b style={{ color: '#0f172a' }}>{stockData.sector}</b> | Kazanan: <b style={{ color: '#0f172a' }}>{stockData.recommendation.best_model}</b>
                          </span>
                        </div>
                        <div style={{ textAlign: 'right' }}>
                          <span style={{ fontSize: 26, fontWeight: 800, color: '#0f172a' }}>{stockData.current_price} ₺</span>
                          <div style={{ fontSize: 11.5, color: stockData.daily_change_pct >= 0 ? '#059669' : '#dc2626', fontWeight: 700 }}>
                            {stockData.daily_change_pct >= 0 ? '+' : ''}%{stockData.daily_change_pct} Günlük
                          </div>
                        </div>
                      </div>

                      <div className="target-numbers-row">
                        <div className="target-pill" style={{ background: '#f0fdf4', borderColor: '#bbf7d0' }}>
                          <div className="target-pill-title" style={{ color: '#166534', fontWeight: 700 }}>Yıl Sonu Hedef Fiyat (12-Ay)</div>
                          <div className="target-pill-val" style={{ color: '#15803d' }}>
                            {stockData.year_end_target || stockData.recommendation.year_end_target || (stockData.current_price * 1.35).toFixed(2)} ₺
                          </div>
                          <div style={{ fontSize: 11, color: '#166534', marginTop: 3, fontWeight: 600 }}>
                            %{stockData.year_end_return_pct >= 0 ? '+' : ''}{stockData.year_end_return_pct || 35.0} Potansiyel
                          </div>
                        </div>
                        <div className="target-pill">
                          <div className="target-pill-title">3 Günlük AI Projeksiyonu</div>
                          <div className="target-pill-val" style={{ color: '#0284c7' }}>
                            {stockData.recommendation.predicted_price} ₺
                          </div>
                          <div style={{ fontSize: 11, color: stockData.recommendation.expected_return_pct >= 0 ? '#059669' : '#dc2626', marginTop: 3, fontWeight: 600 }}>
                            %{stockData.recommendation.expected_return_pct > 0 ? '+' : ''}{stockData.recommendation.expected_return_pct} Kısa Vade
                          </div>
                        </div>
                        <div className="target-pill">
                          <div className="target-pill-title">RSI (14) Momentum</div>
                          <div className="target-pill-val">{stockData.recommendation.indicators.rsi}</div>
                          <div style={{ fontSize: 11, color: '#64748b', marginTop: 3 }}>
                            {stockData.recommendation.indicators.rsi_status?.split(' ')[0] || 'Dengeli'}
                          </div>
                        </div>
                        <div className="target-pill">
                          <div className="target-pill-title">SMA-20 Seviyesi</div>
                          <div className="target-pill-val">{stockData.recommendation.indicators.sma_20} ₺</div>
                          <div style={{ fontSize: 11, color: stockData.current_price >= stockData.recommendation.indicators.sma_20 ? '#059669' : '#dc2626', marginTop: 3 }}>
                            {stockData.current_price >= stockData.recommendation.indicators.sma_20 ? 'Trend Üstü' : 'Trend Altı'}
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="ai-rationale-box">
                      <div style={{ fontWeight: 700, marginBottom: 4, color: '#0f172a', fontSize: 12 }}>
                        CIO ANALİZ RAPORU & ALIM/SATIM GEREKÇESİ:
                      </div>
                      <p>{stockData.recommendation.rationale}</p>
                    </div>
                  </div>
                </div>

                {/* FUTURE 1-2-3 DAY MULTI-MODEL PREDICTION CARDS */}
                {stockData.days_forecast && stockData.days_forecast.length > 0 && (
                  <div style={{ marginTop: 20, marginBottom: 24 }}>
                    <div style={{ fontSize: 13.5, fontWeight: 700, marginBottom: 12, color: '#0f172a', display: 'flex', alignItems: 'center', gap: 6 }}>
                      <Calendar size={16} color="#0284c7" />
                      Gelecek 1, 2 ve 3 Günlük Model Fiyat Projeksiyonları:
                    </div>
                    
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 14 }}>
                      {stockData.days_forecast.map((fc, idx) => {
                        const isPos = fc.change_pct >= 0;
                        return (
                          <div key={idx} style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 10, padding: 16 }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                              <span style={{ fontSize: 13, fontWeight: 800, color: '#0f172a' }}>{fc.day}</span>
                              <span className={`change-badge ${isPos ? 'positive' : 'negative'}`}>
                                {isPos ? '+' : ''}%{fc.change_pct}
                              </span>
                            </div>

                            <div style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: 12.5 }}>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span style={{ color: '#059669', fontWeight: 600 }}>● Attention-GRU:</span>
                                <b style={{ color: '#0f172a' }}>{fc["Attention-GRU"]} ₺</b>
                              </div>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span style={{ color: '#0284c7', fontWeight: 600 }}>● LSTM:</span>
                                <b style={{ color: '#0f172a' }}>{fc["LSTM"]} ₺</b>
                              </div>
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span style={{ color: '#d97706', fontWeight: 600 }}>● GRU:</span>
                                <b style={{ color: '#0f172a' }}>{fc["GRU"]} ₺</b>
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* News Feed */}
                {stockData.news && stockData.news.length > 0 && (
                  <div style={{ marginTop: 24, marginBottom: 20 }}>
                    <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 10, color: '#475569', display: 'flex', alignItems: 'center', gap: 6 }}>
                      <Newspaper size={15} />
                      {stockData.ticker} Canlı Şirket Haberleri:
                    </div>
                    <div className="news-feed-grid">
                      {stockData.news.map((item, idx) => (
                        <a
                          key={idx}
                          href={item.link}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="news-feed-card"
                        >
                          <div className="news-pub">{item.publisher}</div>
                          <div className="news-title">{item.title}</div>
                        </a>
                      ))}
                    </div>
                  </div>
                )}

                {/* Interactive Multi-Model Chart with Future Projection */}
                <div style={{ marginTop: 24 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12, flexWrap: 'wrap', gap: 10 }}>
                    <div>
                      <div style={{ fontSize: 14, fontWeight: 700, color: '#0f172a' }}>
                        {stockData.ticker} Geçmiş Fiyat ve Gelecek 3 Gün Projeksiyon Eğrisi
                      </div>
                      <div style={{ fontSize: 11.5, color: '#64748b' }}>
                        Düz çizgi: Son 20 günün gerçekleşen fiyatı | Kesikli çizgiler: Modellerin gelecek 3 gün tahmini
                      </div>
                    </div>

                    <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
                      {/* Historical Timeframe Switcher */}
                      <div style={{ display: 'flex', gap: 4, background: '#f1f5f9', padding: 3, borderRadius: 6 }}>
                        {[
                          { label: '15G', val: '15' },
                          { label: '30G', val: '30' },
                          { label: '3 Ay', val: '90' },
                          { label: '1 Yıl (Tümü)', val: 'all' }
                        ].map(t => (
                          <button
                            key={t.val}
                            onClick={() => setChartRange(t.val)}
                            style={{
                              border: 'none',
                              background: chartRange === t.val ? '#ffffff' : 'transparent',
                              color: chartRange === t.val ? '#0f172a' : '#64748b',
                              fontSize: 11.5,
                              fontWeight: chartRange === t.val ? 700 : 500,
                              padding: '3px 9px',
                              borderRadius: 4,
                              cursor: 'pointer',
                              boxShadow: chartRange === t.val ? '0 1px 3px rgba(0,0,0,0.08)' : 'none',
                              transition: 'all 0.15s'
                            }}
                          >
                            {t.label}
                          </button>
                        ))}
                      </div>

                      {/* Model Layer Toggles */}
                      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                        <button
                          onClick={() => setShowAttnGRU(!showAttnGRU)}
                          className={`chip-btn ${showAttnGRU ? 'active' : ''}`}
                          style={{ display: 'flex', alignItems: 'center', gap: 5 }}
                        >
                          <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#059669' }}></span>
                          Attention-GRU {showAttnGRU ? '✓' : ''}
                        </button>

                        <button
                          onClick={() => setShowLSTM(!showLSTM)}
                          className={`chip-btn ${showLSTM ? 'active' : ''}`}
                          style={{ display: 'flex', alignItems: 'center', gap: 5 }}
                        >
                          <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#0284c7' }}></span>
                          LSTM {showLSTM ? '✓' : ''}
                        </button>

                        <button
                          onClick={() => setShowGRU(!showGRU)}
                          className={`chip-btn ${showGRU ? 'active' : ''}`}
                          style={{ display: 'flex', alignItems: 'center', gap: 5 }}
                        >
                          <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#d97706' }}></span>
                          GRU {showGRU ? '✓' : ''}
                        </button>

                        <button
                          onClick={() => setShowBacktest(!showBacktest)}
                          className={`chip-btn ${showBacktest ? 'active' : ''}`}
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: 5,
                            borderStyle: 'dashed',
                            borderColor: showBacktest ? '#d97706' : '#cbd5e1'
                          }}
                          title="Modelin geçmiş günlerdeki tahmin eğrisini (backtest) grafikte gösterir/gizler"
                        >
                          <Activity size={12} color={showBacktest ? '#d97706' : '#64748b'} />
                          Geçmiş Model Eğrisi {showBacktest ? '✓' : ''}
                        </button>
                      </div>
                    </div>
                  </div>

                  <div style={{ height: 340, width: '100%' }}>
                    <Line
                      data={getInteractiveChartData()}
                      options={{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { labels: { color: '#475569', font: { size: 12 } } } },
                        scales: {
                          x: { grid: { color: '#f1f5f9' }, ticks: { color: '#64748b' } },
                          y: { grid: { color: '#f1f5f9' }, ticks: { color: '#64748b' } }
                        }
                      }}
                    />
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* TAB 2: FULLY AUTOMATIC OPTIMAL BIST 100 PORTFOLIO */}
      {activeTab === 'portfolio' && (
        <div className="tab-content">
          <div className="glass-panel">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20, flexWrap: 'wrap', gap: 12 }}>
              <div>
                <h2 className="panel-title" style={{ marginBottom: 4 }}>
                  <Award size={20} color="#0284c7" />
                  BIST 100 En Optimal Portföy (Markowitz Max Sharpe)
                </h2>
                <p style={{ fontSize: 13, color: '#64748b' }}>
                  Yapay zeka algoritması Borsa İstanbul piyasasını otomatik tarayarak en yüksek risk-ayarlı getiriyi sağlayan ideal sepeti belirler.
                </p>
              </div>

              <button
                onClick={() => fetchOptimalPortfolio()}
                className="search-btn"
                disabled={loadingPortfolio}
                style={{ padding: '8px 16px', fontSize: 13 }}
              >
                <RefreshCw className={loadingPortfolio ? "animate-spin" : ""} size={14} />
                Tekrar Tara & Optimize Et
              </button>
            </div>

            {loadingPortfolio ? (
              <div style={{ padding: 60, textAlign: 'center', color: '#64748b' }}>
                <RefreshCw className="animate-spin" size={28} style={{ margin: '0 auto 14px auto', color: '#0284c7' }} />
                <div style={{ fontSize: 15, fontWeight: 700, color: '#0f172a', marginBottom: 4 }}>BIST 100 Evreni Taranıyor...</div>
                Bütün lokomotif hisselerin kovaryans ve beklenen getiri matrisleri çözülüyor.
              </div>
            ) : portfolioData && (
              <div>
                {/* Metric Pills */}
                <div className="target-numbers-row" style={{ marginBottom: 24 }}>
                  <div className="target-pill">
                    <div className="target-pill-title">Yıllık Beklenen Getiri</div>
                    <div className="target-pill-val" style={{ color: '#059669' }}>%{portfolioData.annualized_return_pct}</div>
                  </div>
                  <div className="target-pill">
                    <div className="target-pill-title">Portföy Yıllık Riski (Volatilite)</div>
                    <div className="target-pill-val" style={{ color: '#dc2626' }}>%{portfolioData.annualized_volatility_pct}</div>
                  </div>
                  <div className="target-pill">
                    <div className="target-pill-title">Sharpe Oranı (Risk/Getiri)</div>
                    <div className="target-pill-val" style={{ color: '#0284c7' }}>{portfolioData.sharpe_ratio}</div>
                  </div>
                  <div className="target-pill">
                    <div className="target-pill-title">Seçilen Hisse Sayısı</div>
                    <div className="target-pill-val" style={{ color: '#0f172a' }}>{portfolioData.allocations.length} Hisse</div>
                  </div>
                </div>

                <div className="portfolio-grid">
                  <div style={{ height: 320, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <Pie
                      data={getPortfolioPieData()}
                      options={{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { position: 'right', labels: { color: '#475569', font: { size: 12 } } } }
                      }}
                    />
                  </div>

                  <div>
                    <h3 style={{ fontSize: 15, fontWeight: 800, marginBottom: 14, color: '#0f172a' }}>
                      Önerilen Sermaye Dağılımı (% Ağırlıklar)
                    </h3>
                    <div className="allocation-list">
                      {portfolioData.allocations.map((a) => (
                        <div key={a.ticker} className="alloc-item">
                          <div className="alloc-item-left">
                            <span className="alloc-dot" style={{ backgroundColor: a.color }}></span>
                            <div>
                              <div style={{ color: '#0f172a', fontWeight: 800, fontSize: 14 }}>{a.ticker}</div>
                              <div style={{ color: '#64748b', fontSize: 11.5 }}>{a.name}</div>
                            </div>
                          </div>
                          <span style={{ fontWeight: 800, color: '#0f172a', fontSize: 16 }}>%{a.weight}</span>
                        </div>
                      ))}
                    </div>

                    <div className="ai-rationale-box" style={{ marginTop: 18 }}>
                      <div style={{ fontWeight: 700, marginBottom: 6, color: '#0f172a', fontSize: 13, display: 'flex', alignItems: 'center', gap: 6 }}>
                        <ShieldCheck size={16} color="#059669" />
                        CIO Portföy Dağılım Raporu:
                      </div>
                      <p style={{ fontSize: 13.5, color: '#334155' }}>{portfolioData.cio_memo}</p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
