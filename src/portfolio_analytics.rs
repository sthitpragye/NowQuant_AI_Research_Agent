//! # Portfolio Analytics Module
//!
//! Pure Rust, synchronous, no I/O and no LLM calls. This module computes
//! deterministic quant primitives — pairwise return correlation,
//! equal-weighted portfolio volatility, and a diversification ratio — over
//! the price series scraped by `MultiSymbolTool::fetch_basket`.
//!
//! # Design rationale
//!
//! In the multi-agent orchestrator (`orchestrator.rs`), the "quant agent"
//! stage is an LLM call. LLMs are unreliable at exact arithmetic, so we
//! never ask the LLM to compute correlation or volatility itself — we
//! compute it here with normal floating-point math, then hand the LLM a
//! markdown summary of already-correct numbers and ask it only to
//! *interpret* them (concentration risk, what's offsetting what, etc).
//! This keeps the numbers trustworthy regardless of which model is used.

use crate::tools::Candle;
use std::collections::{BTreeSet, HashMap};

/// Aggregate, deterministic risk metrics for a basket of symbols.
#[derive(Debug, Clone)]
pub struct PortfolioRiskSummary {
    /// Symbols that had enough data to be included in the analysis.
    pub symbols: Vec<String>,
    /// Equal weight assigned to each included symbol (1/n).
    pub equal_weight: f64,
    /// Average of all off-diagonal pairwise correlations.
    pub avg_pairwise_correlation: f64,
    /// Full symbol x symbol Pearson correlation matrix (of daily log returns).
    pub correlation_matrix: Vec<Vec<f64>>,
    /// Annualized volatility of an equal-weighted portfolio of all included symbols.
    pub equal_weighted_volatility_pct: f64,
    /// Equal-weighted average of each symbol's own annualized volatility
    /// (i.e. the volatility you'd have with zero diversification benefit).
    pub weighted_avg_individual_volatility_pct: f64,
    /// weighted_avg_individual_volatility_pct / equal_weighted_volatility_pct.
    /// >1 means diversification is reducing portfolio risk below the
    /// average of the individual holdings; ~1 means holdings move together
    /// and diversification isn't helping much.
    pub diversification_ratio: f64,
    /// Number of overlapping trading days used to compute correlations.
    pub aligned_observations: usize,
    /// Symbols excluded for insufficient/errored data.
    pub excluded_symbols: Vec<String>,
}

/// Per-symbol daily log returns, keyed by date string (`YYYY-MM-DD`), so
/// that series from different exchanges/trading calendars can be aligned
/// by intersecting their date keys before computing correlation.
fn daily_log_returns(candles: &[Candle]) -> HashMap<String, f64> {
    let mut map = HashMap::with_capacity(candles.len());
    for i in 1..candles.len() {
        let prev = candles[i - 1].close;
        let curr = candles[i].close;
        if prev > 0.0 && curr > 0.0 {
            map.insert(candles[i].date.clone(), (curr / prev).ln());
        }
    }
    map
}

/// Standard Pearson correlation coefficient between two equal-length series.
fn pearson_correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 2 {
        return 0.0;
    }

    let mean_a = a.iter().sum::<f64>() / n as f64;
    let mean_b = b.iter().sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for i in 0..n {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    if var_a <= 0.0 || var_b <= 0.0 {
        return 0.0;
    }
    cov / (var_a.sqrt() * var_b.sqrt())
}

/// Minimum number of price points a symbol needs to be included.
const MIN_CANDLES: usize = 5;
/// Minimum number of overlapping trading days required to trust correlations.
const MIN_ALIGNED_DAYS: usize = 5;

/// Compute the full portfolio risk summary for a basket of fetched symbol
/// series (as returned by `MultiSymbolTool::fetch_basket`).
///
/// Returns `None` if fewer than two symbols have usable data, or if their
/// trading calendars don't overlap enough to compute a meaningful
/// correlation (e.g. mixing exchanges with very different holidays over a
/// very short range).
pub fn compute_risk_summary(series: &[crate::tools::SymbolSeries]) -> Option<PortfolioRiskSummary> {
    let mut usable: Vec<&crate::tools::SymbolSeries> = series
        .iter()
        .filter(|s| s.error.is_none() && s.stats.is_some() && s.candles.len() >= MIN_CANDLES)
        .collect();
    usable.sort_by(|a, b| a.symbol.cmp(&b.symbol));

    let excluded_symbols: Vec<String> = series
        .iter()
        .filter(|s| s.error.is_some() || s.stats.is_none() || s.candles.len() < MIN_CANDLES)
        .map(|s| s.symbol.clone())
        .collect();

    if usable.len() < 2 {
        return None;
    }

    let return_maps: Vec<HashMap<String, f64>> =
        usable.iter().map(|s| daily_log_returns(&s.candles)).collect();

    let mut common_dates: BTreeSet<String> = return_maps[0].keys().cloned().collect();
    for m in &return_maps[1..] {
        let keys: BTreeSet<String> = m.keys().cloned().collect();
        common_dates = common_dates.intersection(&keys).cloned().collect();
    }
    let common_dates: Vec<String> = common_dates.into_iter().collect();

    if common_dates.len() < MIN_ALIGNED_DAYS {
        return None;
    }

    let returns_matrix: Vec<Vec<f64>> = return_maps
        .iter()
        .map(|m| common_dates.iter().map(|d| *m.get(d).unwrap_or(&0.0)).collect())
        .collect();

    let n = usable.len();
    let mut correlation_matrix = vec![vec![0.0_f64; n]; n];
    let mut pairwise_sum = 0.0;
    let mut pairwise_count = 0usize;

    for i in 0..n {
        for j in 0..n {
            let corr = if i == j {
                1.0
            } else {
                pearson_correlation(&returns_matrix[i], &returns_matrix[j])
            };
            correlation_matrix[i][j] = corr;
            if i < j {
                pairwise_sum += corr;
                pairwise_count += 1;
            }
        }
    }
    let avg_pairwise_correlation = if pairwise_count > 0 {
        pairwise_sum / pairwise_count as f64
    } else {
        0.0
    };

    let equal_weight = 1.0 / n as f64;
    let vols: Vec<f64> = usable
        .iter()
        .map(|s| s.stats.as_ref().unwrap().annualized_volatility_pct / 100.0)
        .collect();

    // Standard two-asset-generalized portfolio variance:
    // sigma_p^2 = sum_i sum_j w_i * w_j * sigma_i * sigma_j * corr_ij
    let mut portfolio_variance = 0.0;
    for i in 0..n {
        for j in 0..n {
            portfolio_variance += equal_weight * equal_weight * vols[i] * vols[j] * correlation_matrix[i][j];
        }
    }
    let equal_weighted_volatility_pct = portfolio_variance.max(0.0).sqrt() * 100.0;
    let weighted_avg_individual_volatility_pct = (vols.iter().sum::<f64>() / n as f64) * 100.0;

    let diversification_ratio = if equal_weighted_volatility_pct > 0.0 {
        weighted_avg_individual_volatility_pct / equal_weighted_volatility_pct
    } else {
        1.0
    };

    Some(PortfolioRiskSummary {
        symbols: usable.iter().map(|s| s.symbol.clone()).collect(),
        equal_weight,
        avg_pairwise_correlation,
        correlation_matrix,
        equal_weighted_volatility_pct,
        weighted_avg_individual_volatility_pct,
        diversification_ratio,
        aligned_observations: common_dates.len(),
        excluded_symbols,
    })
}

impl PortfolioRiskSummary {
    /// Render as a compact markdown block — readable by a human and fed
    /// directly into the quant LLM agent's prompt as ground-truth numbers.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "**Symbols** ({} aligned trading days): {}\n\n",
            self.aligned_observations,
            self.symbols.join(", ")
        ));
        out.push_str(&format!(
            "- Equal-weighted portfolio annualized volatility: {:.2}%\n\
             - Weighted avg. of individual volatilities (no diversification): {:.2}%\n\
             - Diversification ratio: {:.2} (>1 means diversification is reducing risk)\n\
             - Average pairwise correlation: {:.2}\n\n",
            self.equal_weighted_volatility_pct,
            self.weighted_avg_individual_volatility_pct,
            self.diversification_ratio,
            self.avg_pairwise_correlation
        ));

        out.push_str("Correlation matrix (daily log returns):\n\n");
        out.push_str(&format!("| | {} |\n", self.symbols.join(" | ")));
        out.push_str(&format!("|---|{}\n", "---|".repeat(self.symbols.len())));
        for (i, sym) in self.symbols.iter().enumerate() {
            let row: Vec<String> = self.correlation_matrix[i]
                .iter()
                .map(|c| format!("{:.2}", c))
                .collect();
            out.push_str(&format!("| {} | {} |\n", sym, row.join(" | ")));
        }

        if !self.excluded_symbols.is_empty() {
            out.push_str(&format!(
                "\n_Excluded from correlation analysis (insufficient/errored data): {}_\n",
                self.excluded_symbols.join(", ")
            ));
        }

        out
    }
}

// =============================================================================
// UNIT TESTS
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::{FinanceStats, SymbolSeries};

    fn candle(date: &str, close: f64) -> Candle {
        Candle {
            date: date.to_string(),
            open: close,
            high: close,
            low: close,
            close,
            volume: 1000,
        }
    }

    fn stats(symbol: &str, annualized_volatility_pct: f64) -> FinanceStats {
        FinanceStats {
            symbol: symbol.to_string(),
            currency: "USD".to_string(),
            exchange: "TEST".to_string(),
            latest_close: 108.0,
            period_start_close: 100.0,
            period_return_pct: 8.0,
            annualized_volatility_pct,
            max_drawdown_pct: -3.0,
            fifty_two_week_high: None,
            fifty_two_week_low: None,
            avg_daily_volume: 1_000_000,
            num_observations: 6,
        }
    }

    #[test]
    fn test_identical_series_are_perfectly_correlated() {
        let dates = [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
        ];
        let prices = [100.0, 102.0, 101.0, 105.0, 103.0, 108.0];

        let candles_a: Vec<Candle> = dates.iter().zip(prices.iter()).map(|(d, p)| candle(d, *p)).collect();
        let candles_b = candles_a.clone();

        let series = vec![
            SymbolSeries {
                symbol: "A".to_string(),
                candles: candles_a,
                stats: Some(stats("A", 20.0)),
                error: None,
            },
            SymbolSeries {
                symbol: "B".to_string(),
                candles: candles_b,
                stats: Some(stats("B", 20.0)),
                error: None,
            },
        ];

        let summary = compute_risk_summary(&series).expect("expected a risk summary");
        assert!((summary.avg_pairwise_correlation - 1.0).abs() < 1e-6);
        // Identical, fully-correlated, equal-vol assets: portfolio vol == individual vol.
        assert!((summary.equal_weighted_volatility_pct - 20.0).abs() < 1e-3);
        assert!((summary.diversification_ratio - 1.0).abs() < 1e-3);
        assert_eq!(summary.aligned_observations, 5); // 6 prices -> 5 daily returns
    }

    #[test]
    fn test_single_symbol_returns_none() {
        let series = vec![SymbolSeries {
            symbol: "A".to_string(),
            candles: vec![],
            stats: None,
            error: None,
        }];
        assert!(compute_risk_summary(&series).is_none());
    }

    #[test]
    fn test_errored_symbol_is_excluded_not_fatal() {
        let dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"];
        let prices = [100.0, 101.0, 102.0, 100.0, 103.0, 104.0];
        let candles_a: Vec<Candle> = dates.iter().zip(prices.iter()).map(|(d, p)| candle(d, *p)).collect();
        let candles_b = candles_a.clone();

        let series = vec![
            SymbolSeries { symbol: "A".to_string(), candles: candles_a, stats: Some(stats("A", 18.0)), error: None },
            SymbolSeries { symbol: "B".to_string(), candles: candles_b, stats: Some(stats("B", 22.0)), error: None },
            SymbolSeries { symbol: "BADTICKER".to_string(), candles: vec![], stats: None, error: Some("not found".to_string()) },
        ];

        let summary = compute_risk_summary(&series).expect("two good symbols should still produce a summary");
        assert_eq!(summary.symbols.len(), 2);
        assert_eq!(summary.excluded_symbols, vec!["BADTICKER".to_string()]);
    }
}
