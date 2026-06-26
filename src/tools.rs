//! # Tools Module
//!
//! This module implements the web search tool using DuckDuckGo.
//! It demonstrates several important Rust and async patterns:
//! - Trait implementation (Rig's Tool trait)
//! - Async/await for non-blocking I/O
//! - Structured error handling with thiserror
//! - Serde for JSON serialization/deserialization

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;
use tracing::{debug, info, warn};


// =============================================================================
// TAVILY SEARCH TOOL
// =============================================================================

#[derive(Error, Debug)]
pub enum SearchError {
    #[error("Tavily API request failed: {0}")]
    RequestFailed(String),

    #[error("No results returned for query: {0}")]
    NoResults(String),

    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),

    #[error("TAVILY_API_KEY is not set")]
    MissingApiKey,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub content: String,   // full extract, not a meta snippet
    pub score: f64,
}

// Tavily request shape
#[derive(Debug, Serialize)]
struct TavilyRequest<'a> {
    api_key: &'a str,
    query: &'a str,
    search_depth: &'a str,   // "basic" or "advanced"
    max_results: usize,
    include_answer: bool,
}

// Tavily response shape
#[derive(Debug, Deserialize)]
struct TavilyResponse {
    results: Vec<TavilyResult>,
    answer: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TavilyResult {
    title: String,
    url: String,
    content: String,
    score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSearchTool {
    api_key: String,
    max_results: usize,
}

impl WebSearchTool {
    pub fn new(api_key: impl Into<String>, max_results: usize) -> Self {
        Self {
            api_key: api_key.into(),
            max_results,
        }
    }

    pub async fn search(&self, query: &str) -> Result<Vec<SearchResult>, SearchError> {
        if self.api_key.is_empty() {
            return Err(SearchError::MissingApiKey);
        }

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(20))
            .build()?;

        let body = TavilyRequest {
            api_key: &self.api_key,
            query,
            search_depth: "advanced",
            max_results: self.max_results,
            include_answer: true,
        };

        let response = client
            .post("https://api.tavily.com/search")
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(SearchError::RequestFailed(
                format!("HTTP {}", response.status())
            ));
        }

        let data: TavilyResponse = response.json().await
            .map_err(|e| SearchError::RequestFailed(e.to_string()))?;

        if data.results.is_empty() {
            return Err(SearchError::NoResults(query.to_string()));
        }

        Ok(data.results
            .into_iter()
            .map(|r| SearchResult {
                title: r.title,
                url: r.url,
                content: r.content,
                score: r.score,
            })
            .collect())
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SearchArgs {
    pub query: String,
}

impl Tool for WebSearchTool {
    const NAME: &'static str = "web_search";

    type Args = SearchArgs;
    type Output = String;
    type Error = SearchError;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Search the web using Tavily. Returns relevant page extracts \
                          for the query. Use for news, qualitative context, or anything \
                          not covered by the finance tools.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let results = self.search(&args.query).await?;

        let formatted = results
            .iter()
            .enumerate()
            .map(|(i, r)| {
                format!(
                    "{}. **{}** (relevance: {:.2})\n   {}\n   URL: {}\n",
                    i + 1, r.title, r.score, r.content, r.url
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        Ok(format!("## Search Results: {}\n\n{}", args.query, formatted))
    }
}

// =============================================================================
// YAHOO FINANCE TOOL
// =============================================================================
/// This tool scrapes structured market data directly from Yahoo Finance's
/// public (unofficial) "chart" JSON endpoint — the same endpoint Yahoo's own
/// website uses to render quote pages and price charts. No API key is
/// required.
///
/// It works for global tickers, including emerging markets, as long as you
/// use the correct Yahoo suffix, e.g.:
///   - India (NSE):    RELIANCE.NS, TCS.NS
///   - India (BSE):    RELIANCE.BO
///   - Brazil (B3):    PETR4.SA, VALE3.SA
///   - Japan (TSE):    7203.T
///   - Hong Kong:      0700.HK
///   - China (SSE/SZE):600519.SS, 000001.SZ
///   - Indonesia:      BBCA.JK
///   - South Africa:   NPN.JO
///   - Mexico:         WALMEX.MX
///
/// # Rust Concept: Resilient Multi-Host Scraping
///
/// Just like the DuckDuckGo tool above tries multiple HTML-parsing
/// strategies, this tool tries multiple Yahoo Finance hosts (query1/query2)
/// since either can independently rate-limit or fail.
#[derive(Error, Debug)]
pub enum FinanceError {
    #[error("Failed to fetch data from Yahoo Finance: {0}")]
    RequestFailed(String),

    #[error("Failed to parse Yahoo Finance response: {0}")]
    ParseFailed(String),

    #[error("Symbol not found or no data returned for: {0}")]
    SymbolNotFound(String),

    #[error("Rate limited by Yahoo Finance, please wait")]
    RateLimited,

    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),
}

// --- Raw JSON shape returned by query{1,2}.finance.yahoo.com/v8/finance/chart/{symbol} ---

#[derive(Debug, Deserialize)]
struct YahooChartResponse {
    chart: YahooChartWrapper,
}

#[derive(Debug, Deserialize)]
struct YahooChartWrapper {
    result: Option<Vec<YahooChartResult>>,
    error: Option<YahooApiError>,
}

#[derive(Debug, Deserialize)]
struct YahooApiError {
    #[allow(dead_code)]
    code: Option<String>,
    description: Option<String>,
}

#[derive(Debug, Deserialize)]
struct YahooChartResult {
    meta: YahooMeta,
    timestamp: Option<Vec<i64>>,
    indicators: YahooIndicators,
}

/// Metadata block from Yahoo's chart API: current quote snapshot info.
#[derive(Debug, Deserialize, Default, Clone)]
#[serde(rename_all = "camelCase")]
struct YahooMeta {
    currency: Option<String>,
    #[allow(dead_code)]
    symbol: Option<String>,
    exchange_name: Option<String>,
    #[allow(dead_code)]
    instrument_type: Option<String>,
    regular_market_price: Option<f64>,
    previous_close: Option<f64>,
    fifty_two_week_high: Option<f64>,
    fifty_two_week_low: Option<f64>,
    #[allow(dead_code)]
    regular_market_time: Option<i64>,
    long_name: Option<String>,
    short_name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct YahooIndicators {
    quote: Vec<YahooQuoteIndicators>,
}

#[derive(Debug, Deserialize, Default)]
struct YahooQuoteIndicators {
    open: Option<Vec<Option<f64>>>,
    high: Option<Vec<Option<f64>>>,
    low: Option<Vec<Option<f64>>>,
    close: Option<Vec<Option<f64>>>,
    volume: Option<Vec<Option<i64>>>,
}

/// A single OHLCV candle with a human-readable date.
#[derive(Debug, Clone, Serialize)]
pub struct Candle {
    pub date: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: i64,
}

/// Computed quantitative statistics derived from a price series.
/// These are the kinds of metrics a quant research / portfolio analysis
/// agent needs as primitives before any LLM reasoning happens on top.
#[derive(Debug, Clone, Serialize)]
pub struct FinanceStats {
    pub symbol: String,
    pub currency: String,
    pub exchange: String,
    pub latest_close: f64,
    pub period_start_close: f64,
    pub period_return_pct: f64,
    pub annualized_volatility_pct: f64,
    pub max_drawdown_pct: f64,
    pub fifty_two_week_high: Option<f64>,
    pub fifty_two_week_low: Option<f64>,
    pub avg_daily_volume: i64,
    pub num_observations: usize,
}

/// The Yahoo Finance scraping tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YahooFinanceTool {
    /// Default lookback range when the caller doesn't specify one,
    /// e.g. "1d", "5d", "1mo", "3mo", "6mo", "1y", "5y", "ytd", "max".
    default_range: String,

    /// Default candle interval, e.g. "1m", "5m", "15m", "1d", "1wk", "1mo".
    default_interval: String,
}

impl YahooFinanceTool {
    /// Create a new YahooFinanceTool with the given default range/interval.
    ///
    /// # Example
    /// ```
    /// let tool = YahooFinanceTool::new("3mo", "1d");
    /// ```
    pub fn new(default_range: impl Into<String>, default_interval: impl Into<String>) -> Self {
        Self {
            default_range: default_range.into(),
            default_interval: default_interval.into(),
        }
    }

    /// Scrape Yahoo Finance's chart JSON endpoint for a given symbol.
    ///
    /// This hits `https://query{1,2}.finance.yahoo.com/v8/finance/chart/{symbol}`,
    /// which is the unauthenticated JSON feed Yahoo's own front-end uses.
    /// We try query1 first and fall back to query2 if it fails or is
    /// rate-limited, mirroring the resilient multi-strategy approach used
    /// by the web search tool above.
    pub async fn fetch_chart(
        &self,
        symbol: &str,
        range: &str,
        interval: &str,
    ) -> Result<YahooChartResult, FinanceError> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(20))
            .user_agent("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            .build()?;

        let hosts = ["query1.finance.yahoo.com", "query2.finance.yahoo.com"];
        let mut last_err: Option<FinanceError> = None;

        for host in hosts {
            let url = format!(
                "https://{host}/v8/finance/chart/{symbol}?range={range}&interval={interval}&includePrePost=false",
                host = host,
                symbol = urlencoding::encode(symbol),
                range = urlencoding::encode(range),
                interval = urlencoding::encode(interval),
            );

            debug!(url = %url, "Fetching Yahoo Finance chart data");

            let response = match client.get(&url).send().await {
                Ok(r) => r,
                Err(e) => {
                    warn!(host = %host, error = %e, "Yahoo Finance host failed, trying next");
                    last_err = Some(FinanceError::NetworkError(e));
                    continue;
                }
            };

            if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
                warn!(host = %host, "Rate limited by Yahoo Finance");
                last_err = Some(FinanceError::RateLimited);
                continue;
            }

            if !response.status().is_success() {
                last_err = Some(FinanceError::RequestFailed(format!(
                    "HTTP {} from {}",
                    response.status(),
                    host
                )));
                continue;
            }

            let body = match response.text().await {
                Ok(b) => b,
                Err(e) => {
                    last_err = Some(FinanceError::NetworkError(e));
                    continue;
                }
            };

            let parsed: YahooChartResponse = match serde_json::from_str(&body) {
                Ok(p) => p,
                Err(e) => {
                    last_err = Some(FinanceError::ParseFailed(e.to_string()));
                    continue;
                }
            };

            if let Some(api_err) = parsed.chart.error {
                last_err = Some(FinanceError::SymbolNotFound(
                    api_err.description.unwrap_or_else(|| symbol.to_string()),
                ));
                continue;
            }

            match parsed.chart.result.and_then(|mut results| results.pop()) {
                Some(result) => {
                    info!(symbol = %symbol, host = %host, "Yahoo Finance data retrieved");
                    return Ok(result);
                }
                None => {
                    last_err = Some(FinanceError::SymbolNotFound(symbol.to_string()));
                    continue;
                }
            }
        }

        Err(last_err.unwrap_or_else(|| {
            FinanceError::RequestFailed("All Yahoo Finance hosts failed".to_string())
        }))
    }

    /// Convert the raw parallel timestamp/OHLCV arrays Yahoo returns into a
    /// clean, ordered list of candles. Rows missing any OHLC value are
    /// dropped (Yahoo emits nulls for non-trading timestamps).
    pub(crate) fn build_candles(result: &YahooChartResult) -> Vec<Candle> {
        let timestamps = match &result.timestamp {
            Some(t) => t,
            None => return Vec::new(),
        };
        let quote = match result.indicators.quote.first() {
            Some(q) => q,
            None => return Vec::new(),
        };

        let opens = quote.open.as_deref().unwrap_or(&[]);
        let highs = quote.high.as_deref().unwrap_or(&[]);
        let lows = quote.low.as_deref().unwrap_or(&[]);
        let closes = quote.close.as_deref().unwrap_or(&[]);
        let volumes = quote.volume.as_deref().unwrap_or(&[]);

        let mut candles = Vec::with_capacity(timestamps.len());
        for i in 0..timestamps.len() {
            let o = opens.get(i).copied().flatten();
            let h = highs.get(i).copied().flatten();
            let l = lows.get(i).copied().flatten();
            let c = closes.get(i).copied().flatten();

            if let (Some(o), Some(h), Some(l), Some(c)) = (o, h, l, c) {
                let v = volumes.get(i).copied().flatten().unwrap_or(0);
                candles.push(Candle {
                    date: format_timestamp(timestamps[i]),
                    open: o,
                    high: h,
                    low: l,
                    close: c,
                    volume: v,
                });
            }
        }
        candles
    }

    /// Compute summary quantitative statistics from a candle series:
    /// period return, annualized volatility (stdev of log returns scaled by
    /// sqrt(252) trading days), and max drawdown. These are foundational
    /// primitives for downstream strategy refinement / portfolio analysis.
    fn compute_stats(symbol: &str, meta: &YahooMeta, candles: &[Candle]) -> Option<FinanceStats> {
        if candles.is_empty() {
            return None;
        }

        let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
        let latest_close = *closes.last().unwrap();
        let period_start_close = closes[0];

        let period_return_pct = if period_start_close != 0.0 {
            (latest_close / period_start_close - 1.0) * 100.0
        } else {
            0.0
        };

        let mut returns = Vec::with_capacity(closes.len().saturating_sub(1));
        for i in 1..closes.len() {
            if closes[i - 1] > 0.0 && closes[i] > 0.0 {
                returns.push((closes[i] / closes[i - 1]).ln());
            }
        }

        let annualized_volatility_pct = if returns.len() > 1 {
            let mean = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
                / (returns.len() as f64 - 1.0);
            variance.sqrt() * (252.0_f64).sqrt() * 100.0
        } else {
            0.0
        };

        let mut peak = closes[0];
        let mut max_drawdown_pct = 0.0_f64;
        for &c in &closes {
            if c > peak {
                peak = c;
            }
            let drawdown = (c / peak - 1.0) * 100.0;
            if drawdown < max_drawdown_pct {
                max_drawdown_pct = drawdown;
            }
        }

        let avg_daily_volume = candles.iter().map(|c| c.volume).sum::<i64>()
            / candles.len().max(1) as i64;

        Some(FinanceStats {
            symbol: symbol.to_string(),
            currency: meta.currency.clone().unwrap_or_else(|| "N/A".to_string()),
            exchange: meta
                .exchange_name
                .clone()
                .unwrap_or_else(|| "N/A".to_string()),
            latest_close,
            period_start_close,
            period_return_pct,
            annualized_volatility_pct,
            max_drawdown_pct,
            fifty_two_week_high: meta.fifty_two_week_high,
            fifty_two_week_low: meta.fifty_two_week_low,
            avg_daily_volume,
            num_observations: candles.len(),
        })
    }

    /// High-level convenience method: fetch + parse + compute stats in one
    /// call, returning the formatted markdown report. Used both by the
    /// `Tool::call()` implementation below and by callers that want
    /// deterministic finance data without going through the LLM agent.
    pub async fn get_report(
        &self,
        symbol: &str,
        range: Option<&str>,
        interval: Option<&str>,
    ) -> Result<String, FinanceError> {
        let range = range.unwrap_or(&self.default_range);
        let interval = interval.unwrap_or(&self.default_interval);

        let result = self.fetch_chart(symbol, range, interval).await?;
        let candles = Self::build_candles(&result);
        let stats = Self::compute_stats(symbol, &result.meta, &candles);

        Ok(format_finance_report(symbol, &result.meta, &candles, stats.as_ref()))
    }
}

/// Convert a Yahoo Finance unix timestamp (seconds) into a `YYYY-MM-DD` date string.
fn format_timestamp(ts: i64) -> String {
    use chrono::{TimeZone, Utc};
    match Utc.timestamp_opt(ts, 0).single() {
        Some(dt) => dt.format("%Y-%m-%d").to_string(),
        None => ts.to_string(),
    }
}

/// Render the scraped quote + candle + stats data as a markdown report
/// that's easy for both humans and the LLM agent to read.
fn format_finance_report(
    symbol: &str,
    meta: &YahooMeta,
    candles: &[Candle],
    stats: Option<&FinanceStats>,
) -> String {
    let name = meta
        .long_name
        .clone()
        .or_else(|| meta.short_name.clone())
        .unwrap_or_else(|| symbol.to_string());

    let fmt_opt = |v: Option<f64>| v.map(|x| format!("{:.2}", x)).unwrap_or_else(|| "N/A".to_string());

    let mut out = format!("## Yahoo Finance Data: {} ({})\n\n", name, symbol);
    out.push_str(&format!(
        "- **Exchange**: {}\n- **Currency**: {}\n- **Latest Price**: {}\n- **Previous Close**: {}\n- **52-Week Range**: {} – {}\n\n",
        meta.exchange_name.clone().unwrap_or_else(|| "N/A".to_string()),
        meta.currency.clone().unwrap_or_else(|| "N/A".to_string()),
        fmt_opt(meta.regular_market_price),
        fmt_opt(meta.previous_close),
        fmt_opt(meta.fifty_two_week_low),
        fmt_opt(meta.fifty_two_week_high),
    ));

    if let Some(s) = stats {
        out.push_str(&format!(
            "### Quantitative Summary ({} observations)\n- Period Return: {:.2}%\n- Annualized Volatility: {:.2}%\n- Max Drawdown: {:.2}%\n- Avg Daily Volume: {}\n\n",
            s.num_observations,
            s.period_return_pct,
            s.annualized_volatility_pct,
            s.max_drawdown_pct,
            s.avg_daily_volume
        ));
    }

    if candles.is_empty() {
        out.push_str("_No historical candle data was returned for this range/interval._\n");
    } else {
        out.push_str("### Recent Candles (most recent last)\n\n");
        out.push_str("| Date | Open | High | Low | Close | Volume |\n|---|---|---|---|---|---|\n");
        let tail_start = candles.len().saturating_sub(10);
        for c in &candles[tail_start..] {
            out.push_str(&format!(
                "| {} | {:.2} | {:.2} | {:.2} | {:.2} | {} |\n",
                c.date, c.open, c.high, c.low, c.close, c.volume
            ));
        }
    }

    out
}

// =============================================================================
// RIG TOOL TRAIT IMPLEMENTATION (Yahoo Finance)
// =============================================================================
/// Input arguments for the Yahoo Finance tool.
#[derive(Debug, Deserialize, Serialize)]
pub struct FinanceArgs {
    /// Ticker symbol as listed on Yahoo Finance, e.g. "AAPL", "RELIANCE.NS", "PETR4.SA".
    pub symbol: String,

    /// Historical lookback window: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.
    pub range: Option<String>,

    /// Candle interval: 1m, 5m, 15m, 1d, 1wk, 1mo.
    pub interval: Option<String>,
}

impl Tool for YahooFinanceTool {
    const NAME: &'static str = "yahoo_finance";

    type Args = FinanceArgs;
    type Output = String;
    type Error = FinanceError;

    /// Returns the tool definition that describes this tool to the LLM.
    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Scrape real-time quote data and historical OHLCV price history for any \
                ticker symbol directly from Yahoo Finance (no API key required). Supports global \
                markets including emerging markets — use the correct Yahoo suffix, e.g. '.NS'/'.BO' \
                India, '.SA' Brazil, '.T' Japan, '.HK' Hong Kong, '.SS'/'.SZ' China, '.JK' Indonesia, \
                '.JO' South Africa, '.MX' Mexico. Returns latest price, 52-week range, and computed \
                quantitative statistics (period return, annualized volatility, max drawdown) useful \
                for quantitative research, strategy refinement, and portfolio analysis.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Ticker symbol as listed on Yahoo Finance, e.g. AAPL, MSFT, RELIANCE.NS, PETR4.SA, 7203.T"
                    },
                    "range": {
                        "type": "string",
                        "description": "Historical lookback window: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max. Defaults to 3mo."
                    },
                    "interval": {
                        "type": "string",
                        "description": "Candle interval: 1m, 5m, 15m, 1d, 1wk, 1mo. Defaults to 1d."
                    }
                },
                "required": ["symbol"]
            }),
        }
    }

    /// Execute the Yahoo Finance scraping tool.
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.get_report(&args.symbol, args.range.as_deref(), args.interval.as_deref())
            .await
    }
}

// =============================================================================
// SHARED HELPERS (used by all Yahoo-based tools)
// =============================================================================
const YAHOO_USER_AGENT: &str = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36";

// =============================================================================
// FUNDAMENTALS TOOL
// =============================================================================
/// Scrapes company fundamentals (valuation, margins, leverage, sector info)
/// from Yahoo Finance's `quoteSummary` endpoint.
///
/// # Important caveat
///
/// Unlike `v8/finance/chart` (used by `YahooFinanceTool`), Yahoo has
/// progressively tightened anti-scraping protection on `v10/finance/
/// quoteSummary` and now usually requires a session cookie + "crumb" token.
/// This tool fetches that crumb automatically (`fetch_crumb`) using a
/// cookie-enabled client before calling the fundamentals endpoint. If Yahoo
/// changes this flow again, this tool may start failing even though the
/// price/history tool keeps working — that's expected and isolated, since
/// each tool owns its own HTTP logic.
#[derive(Debug, Deserialize, Default, Clone)]
struct RawFmtNumber {
    raw: Option<f64>,
    fmt: Option<String>,
}

impl RawFmtNumber {
    /// Prefer Yahoo's pre-formatted string (e.g. "2.45T", "13.20%") and
    /// fall back to the raw numeric value if no `fmt` was provided.
    fn display(opt: &Option<RawFmtNumber>) -> String {
        match opt {
            Some(v) => v
                .fmt
                .clone()
                .or_else(|| v.raw.map(|r| format!("{:.2}", r)))
                .unwrap_or_else(|| "N/A".to_string()),
            None => "N/A".to_string(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct QuoteSummaryResponse {
    #[serde(rename = "quoteSummary")]
    quote_summary: QuoteSummaryWrapper,
}

#[derive(Debug, Deserialize)]
struct QuoteSummaryWrapper {
    result: Option<Vec<QuoteSummaryResult>>,
    error: Option<YahooApiError>,
}

#[derive(Debug, Deserialize, Default)]
struct QuoteSummaryResult {
    #[serde(rename = "summaryDetail")]
    summary_detail: Option<SummaryDetail>,
    #[serde(rename = "defaultKeyStatistics")]
    default_key_statistics: Option<DefaultKeyStatistics>,
    #[serde(rename = "financialData")]
    financial_data: Option<FinancialData>,
    #[serde(rename = "assetProfile")]
    asset_profile: Option<AssetProfile>,
}

#[derive(Debug, Deserialize, Default)]
struct SummaryDetail {
    #[serde(rename = "marketCap")]
    market_cap: Option<RawFmtNumber>,
    #[serde(rename = "trailingPE")]
    trailing_pe: Option<RawFmtNumber>,
    #[serde(rename = "forwardPE")]
    forward_pe: Option<RawFmtNumber>,
    #[serde(rename = "dividendYield")]
    dividend_yield: Option<RawFmtNumber>,
    beta: Option<RawFmtNumber>,
    #[serde(rename = "averageVolume")]
    average_volume: Option<RawFmtNumber>,
}

#[derive(Debug, Deserialize, Default)]
struct DefaultKeyStatistics {
    #[serde(rename = "profitMargins")]
    profit_margins: Option<RawFmtNumber>,
    #[serde(rename = "pegRatio")]
    peg_ratio: Option<RawFmtNumber>,
    #[serde(rename = "priceToBook")]
    price_to_book: Option<RawFmtNumber>,
    #[serde(rename = "sharesOutstanding")]
    shares_outstanding: Option<RawFmtNumber>,
}

#[derive(Debug, Deserialize, Default)]
struct FinancialData {
    #[serde(rename = "totalRevenue")]
    total_revenue: Option<RawFmtNumber>,
    #[serde(rename = "totalDebt")]
    total_debt: Option<RawFmtNumber>,
    #[serde(rename = "totalCash")]
    total_cash: Option<RawFmtNumber>,
    #[serde(rename = "debtToEquity")]
    debt_to_equity: Option<RawFmtNumber>,
    #[serde(rename = "returnOnEquity")]
    return_on_equity: Option<RawFmtNumber>,
    #[serde(rename = "currentRatio")]
    current_ratio: Option<RawFmtNumber>,
    #[serde(rename = "grossMargins")]
    gross_margins: Option<RawFmtNumber>,
    #[serde(rename = "operatingMargins")]
    operating_margins: Option<RawFmtNumber>,
    #[serde(rename = "revenueGrowth")]
    revenue_growth: Option<RawFmtNumber>,
    #[serde(rename = "recommendationKey")]
    recommendation_key: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct AssetProfile {
    sector: Option<String>,
    industry: Option<String>,
    country: Option<String>,
    #[serde(rename = "fullTimeEmployees")]
    full_time_employees: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct YahooFundamentalsTool;

impl YahooFundamentalsTool {
    pub fn new() -> Self {
        Self
    }

    fn build_client() -> Result<reqwest::Client, FinanceError> {
        Ok(reqwest::Client::builder()
            .timeout(Duration::from_secs(20))
            .user_agent(YAHOO_USER_AGENT)
            .cookie_store(true)
            .build()?)
    }

    /// Seed cookies, then fetch the "crumb" token Yahoo now requires on
    /// `quoteSummary` requests. Best-effort: Yahoo can change this flow.
    async fn fetch_crumb(client: &reqwest::Client) -> Result<String, FinanceError> {
        // Visiting this consent host first seeds the cookies the crumb
        // endpoint expects; ignore failures here, the crumb call below is
        // the one that actually matters.
        let _ = client.get("https://fc.yahoo.com").send().await;

        let resp = client
            .get("https://query2.finance.yahoo.com/v1/test/getcrumb")
            .send()
            .await?;

        if !resp.status().is_success() {
            return Err(FinanceError::RequestFailed(format!(
                "Failed to fetch Yahoo crumb token: HTTP {}",
                resp.status()
            )));
        }

        let crumb = resp.text().await?.trim().to_string();
        if crumb.is_empty() {
            return Err(FinanceError::RequestFailed(
                "Yahoo returned an empty crumb token".to_string(),
            ));
        }
        Ok(crumb)
    }

    async fn fetch_fundamentals(&self, symbol: &str) -> Result<QuoteSummaryResult, FinanceError> {
        let client = Self::build_client()?;
        let crumb = Self::fetch_crumb(&client).await?;

        let hosts = ["query1.finance.yahoo.com", "query2.finance.yahoo.com"];
        let mut last_err: Option<FinanceError> = None;

        for host in hosts {
            let url = format!(
                "https://{host}/v10/finance/quoteSummary/{symbol}?modules=summaryDetail,defaultKeyStatistics,financialData,assetProfile&crumb={crumb}",
                host = host,
                symbol = urlencoding::encode(symbol),
                crumb = urlencoding::encode(&crumb),
            );

            debug!(url = %url, "Fetching Yahoo Finance fundamentals");

            let response = match client.get(&url).send().await {
                Ok(r) => r,
                Err(e) => {
                    last_err = Some(FinanceError::NetworkError(e));
                    continue;
                }
            };

            if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
                last_err = Some(FinanceError::RateLimited);
                continue;
            }
            if !response.status().is_success() {
                last_err = Some(FinanceError::RequestFailed(format!(
                    "HTTP {} from {}",
                    response.status(),
                    host
                )));
                continue;
            }

            let body = match response.text().await {
                Ok(b) => b,
                Err(e) => {
                    last_err = Some(FinanceError::NetworkError(e));
                    continue;
                }
            };

            let parsed: QuoteSummaryResponse = match serde_json::from_str(&body) {
                Ok(p) => p,
                Err(e) => {
                    last_err = Some(FinanceError::ParseFailed(e.to_string()));
                    continue;
                }
            };

            if let Some(api_err) = parsed.quote_summary.error {
                last_err = Some(FinanceError::SymbolNotFound(
                    api_err.description.unwrap_or_else(|| symbol.to_string()),
                ));
                continue;
            }

            match parsed.quote_summary.result.and_then(|mut r| r.pop()) {
                Some(result) => return Ok(result),
                None => {
                    last_err = Some(FinanceError::SymbolNotFound(symbol.to_string()));
                    continue;
                }
            }
        }

        Err(last_err.unwrap_or_else(|| {
            FinanceError::RequestFailed("All Yahoo Finance hosts failed".to_string())
        }))
    }

    fn format_report(symbol: &str, r: &QuoteSummaryResult) -> String {
        let mut out = format!("## Fundamentals: {}\n\n", symbol);

        if let Some(p) = &r.asset_profile {
            out.push_str(&format!(
                "- **Sector**: {}\n- **Industry**: {}\n- **Country**: {}\n- **Employees**: {}\n\n",
                p.sector.clone().unwrap_or_else(|| "N/A".to_string()),
                p.industry.clone().unwrap_or_else(|| "N/A".to_string()),
                p.country.clone().unwrap_or_else(|| "N/A".to_string()),
                p.full_time_employees.map(|n| n.to_string()).unwrap_or_else(|| "N/A".to_string()),
            ));
        }

        out.push_str("### Valuation\n");
        out.push_str(&format!(
            "- Market Cap: {}\n- Trailing P/E: {}\n- Forward P/E: {}\n- PEG Ratio: {}\n- Price/Book: {}\n- Dividend Yield: {}\n- Beta: {}\n\n",
            RawFmtNumber::display(&r.summary_detail.as_ref().and_then(|s| s.market_cap.clone())),
            RawFmtNumber::display(&r.summary_detail.as_ref().and_then(|s| s.trailing_pe.clone())),
            RawFmtNumber::display(&r.summary_detail.as_ref().and_then(|s| s.forward_pe.clone())),
            RawFmtNumber::display(&r.default_key_statistics.as_ref().and_then(|s| s.peg_ratio.clone())),
            RawFmtNumber::display(&r.default_key_statistics.as_ref().and_then(|s| s.price_to_book.clone())),
            RawFmtNumber::display(&r.summary_detail.as_ref().and_then(|s| s.dividend_yield.clone())),
            RawFmtNumber::display(&r.summary_detail.as_ref().and_then(|s| s.beta.clone())),
        ));

        out.push_str("### Profitability & Leverage\n");
        out.push_str(&format!(
            "- Profit Margin: {}\n- Operating Margin: {}\n- Gross Margin: {}\n- Return on Equity: {}\n- Revenue Growth (YoY): {}\n- Debt/Equity: {}\n- Current Ratio: {}\n- Total Revenue: {}\n- Total Debt: {}\n- Total Cash: {}\n- Analyst Recommendation: {}\n",
            RawFmtNumber::display(&r.default_key_statistics.as_ref().and_then(|s| s.profit_margins.clone())),
            RawFmtNumber::display(&r.financial_data.as_ref().and_then(|s| s.operating_margins.clone())),
            RawFmtNumber::display(&r.financial_data.as_ref().and_then(|s| s.gross_margins.clone())),
            RawFmtNumber::display(&r.financial_data.as_ref().and_then(|s| s.return_on_equity.clone())),
            RawFmtNumber::display(&r.financial_data.as_ref().and_then(|s| s.revenue_growth.clone())),
            RawFmtNumber::display(&r.financial_data.as_ref().and_then(|s| s.debt_to_equity.clone())),
            RawFmtNumber::display(&r.financial_data.as_ref().and_then(|s| s.current_ratio.clone())),
            RawFmtNumber::display(&r.financial_data.as_ref().and_then(|s| s.total_revenue.clone())),
            RawFmtNumber::display(&r.financial_data.as_ref().and_then(|s| s.total_debt.clone())),
            RawFmtNumber::display(&r.financial_data.as_ref().and_then(|s| s.total_cash.clone())),
            r.financial_data.as_ref().and_then(|s| s.recommendation_key.clone()).unwrap_or_else(|| "N/A".to_string()),
        ));

        out
    }

    pub async fn get_report(&self, symbol: &str) -> Result<String, FinanceError> {
        let result = self.fetch_fundamentals(symbol).await?;
        Ok(Self::format_report(symbol, &result))
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct FundamentalsArgs {
    /// Ticker symbol, e.g. "AAPL", "RELIANCE.NS", "PETR4.SA".
    pub symbol: String,
}

impl Tool for YahooFundamentalsTool {
    const NAME: &'static str = "yahoo_fundamentals";
    type Args = FundamentalsArgs;
    type Output = String;
    type Error = FinanceError;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Scrape company fundamentals from Yahoo Finance: market cap, P/E (trailing \
                and forward), PEG ratio, price/book, dividend yield, beta, profit/operating/gross \
                margins, return on equity, revenue growth, debt/equity, current ratio, sector, \
                industry, and country. Use this for valuation and quality analysis of a single \
                company, complementing yahoo_finance's price/volatility data.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Ticker symbol as listed on Yahoo Finance, e.g. AAPL, RELIANCE.NS, PETR4.SA"
                    }
                },
                "required": ["symbol"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.get_report(&args.symbol).await
    }
}

// =============================================================================
// FX RATE TOOL
// =============================================================================
/// Currency pair rates via Yahoo Finance. Yahoo represents FX pairs as
/// ordinary chart tickers with a `=X` suffix (e.g. `USDINR=X`, `EURBRL=X`),
/// so this tool is a thin wrapper that builds that symbol and reuses
/// `YahooFinanceTool`'s already-working chart-scraping logic rather than
/// duplicating HTTP/parsing code.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FxRateTool {
    inner: YahooFinanceTool,
}

impl FxRateTool {
    pub fn new() -> Self {
        Self {
            // FX trades continuously across sessions; a short daily window
            // is enough for a "current rate + recent trend" report.
            inner: YahooFinanceTool::new("5d", "1d"),
        }
    }

    fn pair_symbol(from_currency: &str, to_currency: &str) -> String {
        format!(
            "{}{}=X",
            from_currency.trim().to_uppercase(),
            to_currency.trim().to_uppercase()
        )
    }

    pub async fn get_report(&self, from_currency: &str, to_currency: &str) -> Result<String, FinanceError> {
        let symbol = Self::pair_symbol(from_currency, to_currency);
        let report = self.inner.get_report(&symbol, None, None).await?;
        Ok(format!(
            "**FX Pair**: {} → {}\n\n{}",
            from_currency.to_uppercase(),
            to_currency.to_uppercase(),
            report
        ))
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct FxRateArgs {
    /// Base currency code, e.g. "USD".
    pub from_currency: String,
    /// Quote currency code, e.g. "INR".
    pub to_currency: String,
}

impl Tool for FxRateTool {
    const NAME: &'static str = "fx_rate";
    type Args = FxRateArgs;
    type Output = String;
    type Error = FinanceError;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Fetch the current and recent exchange rate between two currencies via \
                Yahoo Finance (e.g. USD/INR, EUR/BRL, USD/ZAR). Essential for normalizing returns \
                across multi-currency, multi-market portfolios.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "from_currency": {
                        "type": "string",
                        "description": "Base/source currency ISO code, e.g. USD"
                    },
                    "to_currency": {
                        "type": "string",
                        "description": "Quote/target currency ISO code, e.g. INR"
                    }
                },
                "required": ["from_currency", "to_currency"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.get_report(&args.from_currency, &args.to_currency).await
    }
}

// =============================================================================
// MULTI-SYMBOL (BATCH/PORTFOLIO) TOOL
// =============================================================================
/// Fetches quote + computed stats for a basket of symbols concurrently and
/// renders a single comparison table. This is the primitive a portfolio
/// analysis agent needs: one call covering every holding instead of N
/// separate lookups.
///
/// # Rust Concept: Concurrent I/O with `futures::future::join_all`
///
/// Each symbol's HTTP fetch is independent, so we issue them all at once
/// and await them together rather than sequentially — much faster for a
/// 10+ symbol portfolio.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiSymbolTool {
    inner: YahooFinanceTool,
}

/// Structured result for one symbol within a basket fetch — the data
/// `MultiSymbolTool::get_report` renders as a table, and what
/// `portfolio_analytics` consumes to compute correlation/portfolio risk.
/// Exposing this (rather than just the markdown string) is what lets the
/// orchestrator's quant stage do real math instead of re-parsing text.
#[derive(Debug, Clone)]
pub struct SymbolSeries {
    pub symbol: String,
    pub candles: Vec<Candle>,
    pub stats: Option<FinanceStats>,
    pub error: Option<String>,
}

impl MultiSymbolTool {
    pub fn new(default_range: impl Into<String>, default_interval: impl Into<String>) -> Self {
        Self {
            inner: YahooFinanceTool::new(default_range, default_interval),
        }
    }

    /// Fetch a basket of symbols concurrently and return structured
    /// per-symbol series (candles + computed stats), rather than a
    /// pre-rendered string. This is the primitive both `get_report` and
    /// the portfolio orchestrator build on.
    pub async fn fetch_basket(
        &self,
        symbols: &[String],
        range: Option<&str>,
        interval: Option<&str>,
    ) -> Vec<SymbolSeries> {
        let range = range.unwrap_or(&self.inner.default_range).to_string();
        let interval = interval.unwrap_or(&self.inner.default_interval).to_string();

        let fetches = symbols.iter().map(|symbol| {
            let inner = self.inner.clone();
            let range = range.clone();
            let interval = interval.clone();
            let symbol = symbol.clone();
            async move {
                let result = inner.fetch_chart(&symbol, &range, &interval).await;
                (symbol, result)
            }
        });

        let results = futures::future::join_all(fetches).await;

        results
            .into_iter()
            .map(|(symbol, fetch_result)| match fetch_result {
                Ok(chart_result) => {
                    let candles = YahooFinanceTool::build_candles(&chart_result);
                    let stats = YahooFinanceTool::compute_stats(&symbol, &chart_result.meta, &candles);
                    SymbolSeries {
                        symbol,
                        candles,
                        stats,
                        error: None,
                    }
                }
                Err(e) => SymbolSeries {
                    symbol,
                    candles: Vec::new(),
                    stats: None,
                    error: Some(e.to_string()),
                },
            })
            .collect()
    }

    pub async fn get_report(
        &self,
        symbols: &[String],
        range: Option<&str>,
        interval: Option<&str>,
    ) -> String {
        let series = self.fetch_basket(symbols, range, interval).await;

        let range = range.unwrap_or(&self.inner.default_range);
        let interval = interval.unwrap_or(&self.inner.default_interval);

        let mut out = format!(
            "## Portfolio Snapshot ({} symbols, range={}, interval={})\n\n",
            symbols.len(),
            range,
            interval
        );
        out.push_str("| Symbol | Currency | Latest Close | Period Return % | Ann. Volatility % | Max Drawdown % |\n");
        out.push_str("|---|---|---|---|---|---|\n");

        for s in &series {
            match (&s.stats, &s.error) {
                (Some(stats), _) => {
                    out.push_str(&format!(
                        "| {} | {} | {:.2} | {:.2} | {:.2} | {:.2} |\n",
                        s.symbol,
                        stats.currency,
                        stats.latest_close,
                        stats.period_return_pct,
                        stats.annualized_volatility_pct,
                        stats.max_drawdown_pct
                    ));
                }
                (None, Some(err)) => {
                    out.push_str(&format!("| {} | - | - | - | - | error: {} |\n", s.symbol, err));
                }
                (None, None) => {
                    out.push_str(&format!("| {} | - | - | - | - | no data |\n", s.symbol));
                }
            }
        }

        out
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct MultiSymbolArgs {
    /// List of ticker symbols to compare, e.g. ["AAPL", "RELIANCE.NS", "PETR4.SA"].
    pub symbols: Vec<String>,
    /// Historical lookback window: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.
    pub range: Option<String>,
    /// Candle interval: 1m, 5m, 15m, 1d, 1wk, 1mo.
    pub interval: Option<String>,
}

impl Tool for MultiSymbolTool {
    const NAME: &'static str = "multi_symbol_snapshot";
    type Args = MultiSymbolArgs;
    type Output = String;
    type Error = FinanceError;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Fetch quote + computed quant stats (return, volatility, drawdown) for a \
                whole basket of tickers concurrently from Yahoo Finance, returned as a single \
                comparison table. Use this for portfolio-level analysis across multiple holdings \
                or markets instead of calling yahoo_finance once per symbol.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "List of ticker symbols, e.g. [\"AAPL\", \"RELIANCE.NS\", \"PETR4.SA\"]"
                    },
                    "range": {
                        "type": "string",
                        "description": "Historical lookback window: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max. Defaults to 3mo."
                    },
                    "interval": {
                        "type": "string",
                        "description": "Candle interval: 1m, 5m, 15m, 1d, 1wk, 1mo. Defaults to 1d."
                    }
                },
                "required": ["symbols"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(self
            .get_report(&args.symbols, args.range.as_deref(), args.interval.as_deref())
            .await)
    }
}

// =============================================================================
// UNIT TESTS
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_web_search_tool_creation() {
        let tool = WebSearchTool::new("test-key", 5);
        assert_eq!(tool.max_results, 5);
    }

    // #[test]
    // fn test_extract_domain() {
    //     assert_eq!(
    //         extract_domain("https://www.example.com/page"),
    //         Some("www.example.com".to_string())
    //     );
    //     assert_eq!(
    //         extract_domain("https://rust-lang.org/learn"),
    //         Some("rust-lang.org".to_string())
    //     );
    // }

    #[test]
    fn test_search_result_serialization() {
        let result = SearchResult {
            title: "Test".to_string(),
            url: "https://test.com".to_string(),
            // snippet: "A test result".to_string(),
            content: "A test result".to_string(),
            score: 0.9,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("Test"));
    }
}
