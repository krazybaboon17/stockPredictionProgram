/**
 * TimesFM Stock Forecasting Platform - Client Application
 */

let currentTicker = "NVDA";
let currentHorizon = 30;
let currentPeriod = "2y";
let currentChartMode = "forecast"; // "forecast", "backtest", "technical"
let currentStockData = null;
let currentForecastData = null;
let currentBacktestData = null;
let currentCompareData = null;
let currentRiskData = null;

let mainChartInstance = null;

let volumeChartInstance = null;

const POPULAR_WATCHLIST = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY", "QQQ"];

document.addEventListener("DOMContentLoaded", () => {
  lucide.createIcons();
  renderWatchlist();
  checkHealth();
  loadStock(currentTicker);
  fetchTasks();
  // Poll tasks every 20 seconds
  setInterval(fetchTasks, 20000);
});

function renderWatchlist() {
  const container = document.getElementById("quickTickers");
  if (!container) return;
  container.innerHTML = POPULAR_WATCHLIST.map(sym => `
    <button onclick="loadStock('${sym}')" class="ticker-pill px-2.5 py-1 rounded-lg bg-surface-850 hover:bg-surface-800 border border-surface-800 hover:border-slate-600 text-slate-300 font-mono transition-all flex items-center gap-1.5 ${sym === currentTicker ? 'border-brand-500/70 bg-brand-500/10 text-white font-bold' : ''}">
      <span>${sym}</span>
    </button>
  `).join("");
}

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    document.getElementById("modelStatusText").textContent = data.model;
  } catch (e) {
    document.getElementById("modelStatusText").textContent = "Service Connecting...";
  }
}

async function refreshAll(force = false) {
  const icon = document.getElementById("refreshIcon");
  if (icon) icon.classList.add("animate-spin");
  await loadStock(currentTicker, force);
  await fetchTasks();
  if (icon) setTimeout(() => icon.classList.remove("animate-spin"), 600);
}

function setHorizon(days) {
  currentHorizon = days;
  document.querySelectorAll(".horizon-btn").forEach(btn => {
    if (parseInt(btn.getAttribute("data-h")) === days) {
      btn.className = "horizon-btn px-3 py-1 rounded-lg text-xs font-mono font-bold transition-all bg-brand-600 text-white";
    } else {
      btn.className = "horizon-btn px-3 py-1 rounded-lg text-xs font-mono font-medium transition-all text-slate-400 hover:text-white";
    }
  });
  document.getElementById("targetHorizonDays").textContent = `${days} Days`;
  document.getElementById("quantileHorizonDay").textContent = `${days}`;
  fetchPrediction();
  fetchRiskMetrics();
}



function setChartMode(mode) {
  currentChartMode = mode;
  const modes = {
    forecast: document.getElementById("modeForecastBtn"),
    compare: document.getElementById("modeCompareBtn"),
    backtest: document.getElementById("modeBacktestBtn"),
    technical: document.getElementById("modeTechBtn")
  };

  Object.keys(modes).forEach(k => {
    if (modes[k]) {
      if (k === mode) {
        modes[k].className = "px-3 py-1 rounded-lg text-xs font-medium bg-brand-600 text-white transition-all flex items-center gap-1.5";
      } else {
        modes[k].className = "px-3 py-1 rounded-lg text-xs font-medium text-slate-400 hover:text-white transition-all flex items-center gap-1.5";
      }
    }
  });

  const volContainer = document.getElementById("volumeSubplotContainer");
  const compareLeaderboard = document.getElementById("compareLeaderboard");

  if (mode === "compare") {
    if (volContainer) volContainer.classList.add("hidden");
    if (compareLeaderboard) compareLeaderboard.classList.remove("hidden");
    fetchComparison();
  } else {
    if (volContainer) volContainer.classList.remove("hidden");
    if (compareLeaderboard) compareLeaderboard.classList.add("hidden");
    renderChart();
  }
}

function exportData(format) {
  const url = `/api/export/${currentTicker}?format=${format}&horizon=${currentHorizon}&period=${currentPeriod}`;
  window.open(url, "_blank");
}

async function fetchComparison() {
  const overlay = document.getElementById("chartLoadingOverlay");
  if (overlay) overlay.classList.remove("hidden");

  try {
    const res = await fetch("/api/predict/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        tickers: ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
        horizon: currentHorizon,
        period: currentPeriod
      })
    });
    if (!res.ok) throw new Error("Comparative forecast request failed");
    currentCompareData = await res.json();

    const tbody = document.getElementById("compareTableBody");
    if (tbody && currentCompareData.comparisons) {
      tbody.innerHTML = currentCompareData.comparisons.map((c, i) => `
        <tr class="hover:bg-surface-800/40 transition-colors">
          <td class="py-2.5 px-3 font-bold text-white flex items-center gap-2">
            <span class="w-2.5 h-2.5 rounded-full ${i === 0 ? 'bg-amber-400 ring-2 ring-amber-400/30' : 'bg-slate-500'}"></span>
            <span>${c.ticker}</span>
            <span class="text-[10px] text-slate-400 font-normal">(${c.name})</span>
          </td>
          <td class="py-2.5 px-3 text-slate-300 font-mono">$${c.start_price.toFixed(2)}</td>
          <td class="py-2.5 px-3 font-semibold text-accent-cyan font-mono">$${c.predicted_end_price.toFixed(2)}</td>
          <td class="py-2.5 px-3 font-bold font-mono ${c.expected_return_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}">
            ${c.expected_return_pct >= 0 ? '+' : ''}${c.expected_return_pct.toFixed(2)}%
          </td>
          <td class="py-2.5 px-3">
            <span class="px-2 py-0.5 rounded text-[10px] font-bold font-mono ${c.trend === 'BULLISH' ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30' : (c.trend === 'BEARISH' ? 'bg-rose-500/10 text-rose-400 border border-rose-500/30' : 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/30')}">
              ${c.trend}
            </span>
          </td>
        </tr>
      `).join("");
    }
    renderChart();
  } catch (err) {
    console.error("Comparison fetch failed:", err);
  } finally {
    if (overlay) overlay.classList.add("hidden");
    lucide.createIcons();
  }
}

async function loadStock(ticker, forceRefresh = false) {
  const inputEl = document.getElementById("tickerInput");
  const target = (ticker || (inputEl ? inputEl.value : "") || "NVDA").trim().toUpperCase();
  if (!target) {
    alert("Please enter a valid stock ticker symbol.");
    return;
  }
  currentTicker = target;
  renderWatchlist();

  const overlay = document.getElementById("chartLoadingOverlay");
  if (overlay) overlay.classList.remove("hidden");

  try {
    const stockRes = await fetch(`/api/stock/${currentTicker}?period=${currentPeriod}&force_refresh=${forceRefresh}`);
    if (!stockRes.ok) {
      const err = await stockRes.json();
      throw new Error(err.detail || "Failed to load stock data");
    }
    currentStockData = await stockRes.json();

    // Update Overview details
    document.getElementById("stockSymbol").textContent = currentStockData.ticker;
    document.getElementById("stockName").textContent = currentStockData.name;
    document.getElementById("stockSector").textContent = currentStockData.sector;
    document.getElementById("currentPrice").textContent = `$${currentStockData.current_price.toFixed(2)}`;

    const change = currentStockData.change;
    const changePct = currentStockData.change_percent;
    const isUp = change >= 0;
    const badge = document.getElementById("priceChangeBadge");
    if (badge) {
      badge.className = `flex items-center gap-1 text-sm font-semibold font-mono px-2 py-0.5 rounded-lg ${isUp ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'}`;
      badge.innerHTML = `<i data-lucide="${isUp ? 'arrow-up-right' : 'arrow-down-right'}" class="w-4 h-4"></i><span>${isUp ? '+' : ''}${change.toFixed(2)} (${isUp ? '+' : ''}${changePct.toFixed(2)}%)</span>`;
    }

    // Fundamentals
    document.getElementById("statMarketCap").textContent = currentStockData.market_cap ? `$${(currentStockData.market_cap / 1e9).toFixed(1)}B` : '--';
    document.getElementById("statPE").textContent = currentStockData.pe_ratio ? currentStockData.pe_ratio.toFixed(1) : '24.2';
    document.getElementById("stat52High").textContent = currentStockData.week_52_high ? `$${currentStockData.week_52_high.toFixed(2)}` : '--';
    document.getElementById("stat52Low").textContent = currentStockData.week_52_low ? `$${currentStockData.week_52_low.toFixed(2)}` : '--';
    document.getElementById("statPoints").textContent = currentStockData.data_points || '--';

    // Fetch TimesFM Forecast, Backtest & Risk Metrics concurrently
    await Promise.all([
      fetchPrediction(),
      fetchBacktest(),
      fetchRiskMetrics()
    ]);

  } catch (err) {
    console.error("Failed loading stock:", err);
    alert(`Notice: ${err.message}`);
  } finally {
    if (overlay) overlay.classList.add("hidden");
    lucide.createIcons();
  }
}

async function fetchPrediction() {
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker: currentTicker, horizon: currentHorizon, period: currentPeriod })
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Prediction request failed");
    }
    const json = await res.json();
    currentForecastData = json.forecast;

    // Update forecast highlight banner
    const targetPrice = currentForecastData.predicted_end_price;
    const targetChangePct = currentForecastData.change_percent;
    const isUp = targetChangePct >= 0;

    document.getElementById("predictedTargetPrice").textContent = `$${targetPrice.toFixed(2)}`;
    const pctEl = document.getElementById("predictedChangePct");
    pctEl.textContent = `${isUp ? '+' : ''}${targetChangePct.toFixed(2)}%`;
    pctEl.className = `text-xs font-semibold font-mono ${isUp ? 'text-emerald-400' : 'text-rose-400'}`;

    const trendBadge = document.getElementById("predictedTrendBadge");
    trendBadge.textContent = currentForecastData.trend;
    trendBadge.className = `font-mono font-bold ${currentForecastData.trend === 'BULLISH' ? 'text-emerald-400' : (currentForecastData.trend === 'BEARISH' ? 'text-rose-400' : 'text-cyan-400')}`;

    // Quantiles for last prediction step
    const lastStep = currentForecastData.predictions[currentForecastData.predictions.length - 1];
    if (lastStep) {
      document.getElementById("valQ90").textContent = `$${lastStep.q90.toFixed(2)}`;
      document.getElementById("valQ75").textContent = `$${lastStep.q75.toFixed(2)}`;
      document.getElementById("valQ50").textContent = `$${lastStep.q50.toFixed(2)}`;
      document.getElementById("valQ25").textContent = `$${lastStep.q25.toFixed(2)}`;
      document.getElementById("valQ10").textContent = `$${lastStep.q10.toFixed(2)}`;
    }

    renderChart();
  } catch (e) {
    console.error("Prediction fetch error:", e);
  }
}

async function fetchBacktest() {
  try {
    const res = await fetch("/api/backtest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker: currentTicker, test_days: 30, period: currentPeriod })
    });
    if (!res.ok) return;
    const json = await res.json();
    currentBacktestData = json.backtest;

    document.getElementById("metricMAE").textContent = `$${currentBacktestData.mae.toFixed(2)}`;
    document.getElementById("metricRMSE").textContent = `$${currentBacktestData.rmse.toFixed(2)}`;
    document.getElementById("metricMAPE").textContent = `${currentBacktestData.mape.toFixed(2)}%`;
    document.getElementById("metricDirAcc").textContent = `${currentBacktestData.directional_accuracy_pct.toFixed(1)}%`;
  } catch (e) {
    console.error("Backtest fetch error:", e);
  }
}

async function fetchRiskMetrics() {
  try {
    const res = await fetch(`/api/risk/${currentTicker}?horizon=${currentHorizon}&period=${currentPeriod}`);
    if (!res.ok) return;
    const json = await res.json();
    const risk = json.risk_metrics;
    if (!risk) return;

    const var95El = document.getElementById("metricVaR95");
    const cvar95El = document.getElementById("metricCVaR95");
    const volEl = document.getElementById("metricAnnualVol");
    const sharpeEl = document.getElementById("metricSharpe");
    const riskTierEl = document.getElementById("metricRiskTier") || document.getElementById("riskTierBadge");

    if (var95El) var95El.textContent = `${risk.var_95_pct}% ($${risk.var_95_amount.toFixed(2)})`;
    if (cvar95El) cvar95El.textContent = `${risk.cvar_95_pct}% ($${risk.cvar_95_amount.toFixed(2)})`;
    if (volEl) volEl.textContent = `${risk.annualized_volatility_pct}%`;
    if (sharpeEl) sharpeEl.textContent = `${risk.sharpe_ratio >= 0 ? '+' : ''}${risk.sharpe_ratio.toFixed(2)}`;
    if (riskTierEl) {
      riskTierEl.textContent = risk.risk_tier;
      riskTierEl.className = `font-mono font-bold text-[10px] px-2 py-0.5 rounded bg-${risk.risk_color}-500/10 text-${risk.risk_color}-400 border border-${risk.risk_color}-500/30`;
    }

  } catch (e) {
    console.error("Risk metrics error:", e);
  }
}

function updateChartVisibility() {
  renderChart();
}

function renderChart() {
  if (!currentStockData || !currentStockData.history) return;

  const history = currentStockData.history;
  const historyLabels = history.map(h => h.date);
  const closePrices = history.map(h => h.close);

  // Latest technical values for gauges
  const latestHist = history[history.length - 1];
  if (latestHist) {
    const rsiVal = latestHist.rsi || 50;
    document.getElementById("rsiValueText").textContent = rsiVal.toFixed(1);
    document.getElementById("rsiBar").style.width = `${Math.min(100, Math.max(0, rsiVal))}%`;

    const macdHist = latestHist.macd_hist || 0;
    const macdEl = document.getElementById("macdHistVal");
    macdEl.textContent = `${macdHist >= 0 ? '+' : ''}${macdHist.toFixed(3)}`;
    macdEl.className = `font-bold ${macdHist >= 0 ? 'text-emerald-400' : 'text-rose-400'}`;

    document.getElementById("bbSpreadVal").textContent = `$${(latestHist.bb_upper || 0).toFixed(1)} / $${(latestHist.bb_lower || 0).toFixed(1)}`;
  }

  // Check toggles
  const showSMA = document.getElementById("toggleSMA").checked;
  const showBB = document.getElementById("toggleBB").checked;
  const showQuantiles = document.getElementById("toggleQuantiles").checked;

  let allLabels = [];
  let datasets = [];

  if (currentChartMode === "forecast") {
    document.getElementById("chartTitle").innerHTML = `<i data-lucide="sparkles" class="w-4 h-4 text-cyan-400"></i> TimesFM Zero-Shot Forecast Trajectory`;
    document.getElementById("chartSubtitle").textContent = `Context Window: 252 Days • Multi-Horizon Quantile Bands (10th - 90th)`;

    const windowSlice = Math.min(120, history.length);
    const slicedHistory = history.slice(-windowSlice);
    const slicedLabels = slicedHistory.map(h => h.date);
    const slicedClose = slicedHistory.map(h => h.close);

    const forecastPts = currentForecastData ? currentForecastData.predictions : [];
    const forecastLabels = forecastPts.map(p => p.date);

    allLabels = [...slicedLabels, ...forecastLabels];

    // 1. Historical Line
    const histDataPadded = [...slicedClose, ...forecastLabels.map(() => null)];
    datasets.push({
      label: "Historical Close",
      data: histDataPadded,
      borderColor: "#94a3b8",
      backgroundColor: "#94a3b8",
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.1,
      order: 3
    });

    if (showSMA) {
      const sma20Padded = [...slicedHistory.map(h => h.sma_20), ...forecastLabels.map(() => null)];
      const sma50Padded = [...slicedHistory.map(h => h.sma_50), ...forecastLabels.map(() => null)];
      datasets.push({
        label: "SMA 20",
        data: sma20Padded,
        borderColor: "rgba(245, 158, 11, 0.7)",
        borderWidth: 1.5,
        borderDash: [4, 4],
        pointRadius: 0,
        tension: 0.1,
        order: 4
      });
      datasets.push({
        label: "SMA 50",
        data: sma50Padded,
        borderColor: "rgba(168, 85, 247, 0.7)",
        borderWidth: 1.5,
        borderDash: [5, 5],
        pointRadius: 0,
        tension: 0.1,
        order: 5
      });
    }

    if (showBB) {
      const bbUpPadded = [...slicedHistory.map(h => h.bb_upper), ...forecastLabels.map(() => null)];
      const bbLowPadded = [...slicedHistory.map(h => h.bb_lower), ...forecastLabels.map(() => null)];
      datasets.push({
        label: "Bollinger Upper",
        data: bbUpPadded,
        borderColor: "rgba(56, 189, 248, 0.35)",
        borderWidth: 1,
        pointRadius: 0,
        tension: 0.1,
        order: 6
      });
      datasets.push({
        label: "Bollinger Lower",
        data: bbLowPadded,
        borderColor: "rgba(56, 189, 248, 0.35)",
        borderWidth: 1,
        pointRadius: 0,
        tension: 0.1,
        order: 7
      });
    }

    // TimesFM Forecast datasets
    if (forecastPts.length > 0) {
      const lastClose = slicedClose[slicedClose.length - 1];
      const offset = slicedLabels.length - 1;

      const createForecastSeries = (valExtractor) => {
        const arr = new Array(allLabels.length).fill(null);
        arr[offset] = lastClose; // anchor
        forecastPts.forEach((pt, i) => {
          arr[offset + 1 + i] = valExtractor(pt);
        });
        return arr;
      };

      const q90Data = createForecastSeries(p => p.q90);
      const q75Data = createForecastSeries(p => p.q75);
      const q50Data = createForecastSeries(p => p.predicted);
      const q25Data = createForecastSeries(p => p.q25);
      const q10Data = createForecastSeries(p => p.q10);

      if (showQuantiles) {
        datasets.push({
          label: "80% CI Upper (q90)",
          data: q90Data,
          borderColor: "transparent",
          backgroundColor: "rgba(6, 182, 212, 0.08)",
          pointRadius: 0,
          fill: "+4",
          tension: 0.2,
          order: 8
        });

        datasets.push({
          label: "50% CI Upper (q75)",
          data: q75Data,
          borderColor: "transparent",
          backgroundColor: "rgba(6, 182, 212, 0.16)",
          pointRadius: 0,
          fill: "+2",
          tension: 0.2,
          order: 9
        });
      }

      datasets.push({
        label: "TimesFM Median Forecast",
        data: q50Data,
        borderColor: "#06b6d4",
        backgroundColor: "#06b6d4",
        borderWidth: 2.5,
        pointRadius: 2,
        pointHoverRadius: 5,
        tension: 0.2,
        order: 1
      });

      if (showQuantiles) {
        datasets.push({
          label: "50% CI Lower (q25)",
          data: q25Data,
          borderColor: "transparent",
          pointRadius: 0,
          tension: 0.2,
          order: 10
        });
        datasets.push({
          label: "80% CI Lower (q10)",
          data: q10Data,
          borderColor: "transparent",
          pointRadius: 0,
          tension: 0.2,
          order: 11
        });
      }
    }

  } else if (currentChartMode === "backtest") {
    document.getElementById("chartTitle").innerHTML = `<i data-lucide="check-circle-2" class="w-4 h-4 text-emerald-400"></i> Out-of-Sample Historical Backtest`;
    document.getElementById("chartSubtitle").textContent = `TimesFM evaluated on held-out 30-day window against historical truth`;

    if (currentBacktestData && currentBacktestData.comparison) {
      const comp = currentBacktestData.comparison;
      allLabels = comp.map(c => c.date);
      const actuals = comp.map(c => c.actual);
      const preds = comp.map(c => c.predicted);

      datasets.push({
        label: "Ground Truth Actual",
        data: actuals,
        borderColor: "#10b981",
        backgroundColor: "#10b981",
        borderWidth: 2.5,
        pointRadius: 3,
        tension: 0.2
      });

      datasets.push({
        label: "TimesFM Prediction",
        data: preds,
        borderColor: "#06b6d4",
        backgroundColor: "#06b6d4",
        borderWidth: 2,
        borderDash: [4, 4],
        pointRadius: 3,
        tension: 0.2
      });
    }

  } else if (currentChartMode === "technical") {
    document.getElementById("chartTitle").innerHTML = `<i data-lucide="sliders" class="w-4 h-4 text-purple-400"></i> Technical Indicator Overlay`;
    document.getElementById("chartSubtitle").textContent = `SMA 20/50/200, Bollinger Bands and Price Envelope`;

    const windowSlice = Math.min(180, history.length);
    const sliced = history.slice(-windowSlice);
    allLabels = sliced.map(h => h.date);

    datasets.push({
      label: "Price Close",
      data: sliced.map(h => h.close),
      borderColor: "#f8fafc",
      borderWidth: 2,
      pointRadius: 0
    });
    datasets.push({
      label: "SMA 20",
      data: sliced.map(h => h.sma_20),
      borderColor: "#f59e0b",
      borderWidth: 1.5,
      pointRadius: 0
    });
    datasets.push({
      label: "SMA 50",
      data: sliced.map(h => h.sma_50),
      borderColor: "#a855f7",
      borderWidth: 1.5,
      pointRadius: 0
    });
    datasets.push({
      label: "Bollinger Upper",
      data: sliced.map(h => h.bb_upper),
      borderColor: "rgba(56, 189, 248, 0.4)",
      borderWidth: 1,
      pointRadius: 0
    });
    datasets.push({
      label: "Bollinger Lower",
      data: sliced.map(h => h.bb_lower),
      borderColor: "rgba(56, 189, 248, 0.4)",
      borderWidth: 1,
      pointRadius: 0
    });
  } else if (currentChartMode === "compare") {
    document.getElementById("chartTitle").innerHTML = `<i data-lucide="layers" class="w-4 h-4 text-cyan-400"></i> TimesFM Multi-Asset Normalized Return Forecast (%)`;
    document.getElementById("chartSubtitle").textContent = `Comparison baseline normalized to 0.0% at Day 0 • Horizon: ${currentHorizon} Days`;

    if (currentCompareData && currentCompareData.comparisons && currentCompareData.comparisons.length > 0) {
      allLabels = ["Today", ...currentCompareData.dates];
      const palette = ["#06b6d4", "#10b981", "#f59e0b", "#a855f7", "#f43f5e", "#3b82f6", "#ec4899", "#84cc16"];

      currentCompareData.comparisons.forEach((item, idx) => {
        const color = palette[idx % palette.length];
        const trajectory = [0.0, ...item.normalized_pct_trajectory];
        const isUp = item.expected_return_pct >= 0;
        datasets.push({
          label: `${item.ticker} (${isUp ? '+' : ''}${item.expected_return_pct.toFixed(1)}%)`,
          data: trajectory,
          borderColor: color,
          backgroundColor: color,
          borderWidth: 2,
          pointRadius: 2,
          pointHoverRadius: 5,
          tension: 0.2
        });
      });
    }
  }

  // Draw Main Chart
  const ctx = document.getElementById("mainChart").getContext("2d");
  if (mainChartInstance) mainChartInstance.destroy();

  const isCompareMode = currentChartMode === "compare";

  mainChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: allLabels,
      datasets: datasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index',
        intersect: false,
      },
      plugins: {
        legend: { display: isCompareMode, labels: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 11 } } },
        tooltip: {
          backgroundColor: '#0d111a',
          titleColor: '#e2e8f0',
          bodyColor: '#cbd5e1',
          borderColor: '#2b3952',
          borderWidth: 1,
          padding: 10,
          displayColors: true,
          callbacks: {
            label: function(context) {
              let label = context.dataset.label || '';
              if (label) label += ': ';
              if (context.parsed.y !== null) {
                if (isCompareMode) {
                  label += (context.parsed.y >= 0 ? '+' : '') + context.parsed.y.toFixed(2) + '%';
                } else {
                  label += '$' + context.parsed.y.toFixed(2);
                }
              }
              return label;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: {
            color: '#64748b',
            font: { family: 'JetBrains Mono', size: 10 },
            maxTicksLimit: 12
          }
        },
        y: {
          position: 'right',
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: {
            color: '#64748b',
            font: { family: 'JetBrains Mono', size: 10 },
            callback: value => isCompareMode ? (value >= 0 ? '+' : '') + value.toFixed(1) + '%' : '$' + value.toFixed(1)
          }
        }
      }
    }
  });

  // Draw Volume Subplot
  const vCtx = document.getElementById("volumeChart").getContext("2d");
  if (volumeChartInstance) volumeChartInstance.destroy();

  const volWindow = Math.min(120, history.length);
  const volSlice = history.slice(-volWindow);
  volumeChartInstance = new Chart(vCtx, {
    type: 'bar',
    data: {
      labels: volSlice.map(h => h.date),
      datasets: [{
        label: "Volume",
        data: volSlice.map(h => h.volume),
        backgroundColor: volSlice.map(h => h.close >= h.open ? 'rgba(16, 185, 129, 0.35)' : 'rgba(244, 63, 94, 0.35)'),
        borderWidth: 0,
        borderRadius: 2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { enabled: false }
      },
      scales: {
        x: { display: false },
        y: { display: false }
      }
    }
  });

  lucide.createIcons();
}

// --- Agent Tasks Coordination Functions ---
async function fetchTasks() {
  try {
    const res = await fetch("/api/tasks");
    const json = await res.json();
    const content = json.content || "TASKS.txt is empty.";
    const tb = document.getElementById("tasksBox");
    const dt = document.getElementById("drawerFullTasksText");
    if (tb) tb.textContent = content;
    if (dt) dt.textContent = content;
  } catch (err) {
    const tb = document.getElementById("tasksBox");
    if (tb) tb.textContent = "Could not reach TASKS.txt API.";
  }
}

async function submitTask() {
  const descInput = document.getElementById("newTaskDesc");
  const priority = document.getElementById("newTaskPriority").value;
  const desc = descInput.value.trim();
  if (!desc) return;

  try {
    await fetch("/api/tasks/create", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        from_role: "Tester/PM",
        to_role: "Lead Developer",
        priority: priority,
        description: desc
      })
    });
    descInput.value = "";
    await fetchTasks();
  } catch (err) {
    alert("Failed to submit task to TASKS.txt");
  }
}

function toggleTaskDrawer() {
  const drawer = document.getElementById("taskDrawerModal");
  if (drawer) drawer.classList.toggle("hidden");
}
