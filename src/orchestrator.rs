//! # Portfolio Orchestrator Module
//!
//! Rig — like most lightweight Rust agent frameworks — doesn't ship a
//! built-in multi-agent graph/planner the way LangGraph or AutoGen do in
//! the Python ecosystem. This module is the manual equivalent: a fixed,
//! auditable pipeline of specialized LLM calls, each with its own narrow
//! preamble and responsibility, where the output of one stage becomes the
//! input to the next.
//!
//! ## Pipeline stages
//! 1. **Data gather** (deterministic, no LLM) — `MultiSymbolTool::fetch_basket`
//! 2. **Quant interpretation** (LLM, no tools) — reads pre-computed risk numbers
//! 3. **Market research** (LLM + `web_search` tool) — qualitative context
//! 4. **Portfolio advice** (LLM, no tools) — synthesizes stages 2 + 3
//! 5. **Final report** (LLM, no tools) — structured write-up of everything
//!
//! ## Feasibility note
//! Splitting responsibilities into separate, narrowly-scoped LLM calls
//! (instead of one agent juggling five tools and five jobs at once) keeps
//! behavior predictable and makes each stage independently testable and
//! swappable. The tradeoff versus a framework like AutoGen/LangGraph is
//! that *we* own the control flow explicitly — there's no planner deciding
//! the stage order at runtime, which is exactly what makes this approach
//! reliable enough for financial reporting.
//!
//! ## Why quant math never touches the LLM
//! Stage 1 and the correlation/volatility math in `portfolio_analytics`
//! are pure Rust. The quant LLM agent in stage 2 is only ever asked to
//! *interpret* numbers we've already computed correctly — never to
//! calculate them — because LLM arithmetic isn't trustworthy enough for
//! risk figures that might end up in a real report.

use anyhow::Result;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::ollama;
use tracing::info;

use crate::config::Config;
use crate::portfolio_analytics;
use crate::tools::{MultiSymbolTool, WebSearchTool};

const QUANT_AGENT_PREAMBLE: &str = r#"
You are a quantitative risk analyst. You will be given pre-computed,
already-correct numbers: a correlation matrix, portfolio volatility, and a
diversification ratio for a basket of holdings. Do NOT recompute or
second-guess the numbers - treat them as ground truth. Your job is only to
INTERPRET them: call out concentration risk, which pairs are most/least
correlated and what that implies, and whether the diversification ratio
suggests this basket is well diversified or not. Be concise (150-250 words),
no preamble, do not repeat the raw table back verbatim.
"#;

const RESEARCH_AGENT_PREAMBLE: &str = r#"
You are a markets research assistant. Use the web_search tool ONCE to find
current, relevant context for the given symbols/topic, then summarize
concisely (150-250 words) with sources. Never call the tool more than once
per turn.
"#;

const PORTFOLIO_AGENT_PREAMBLE: &str = r#"
You are a portfolio strategist. You will be given (1) a quant analyst's risk
interpretation of a basket of holdings and (2) qualitative market research
context. Combine them into concrete, actionable portfolio observations:
where concentration risk could be trimmed, what diversification gaps exist,
and what an investor might watch for next. Include one brief line noting
this isn't personalized financial advice. Be concise (200-300 words).
"#;

const REPORT_AGENT_PREAMBLE: &str = r#"
You are a financial report writer. You will be given a quant risk
interpretation, market research context, and portfolio strategy notes for a
basket of holdings. Combine them into ONE polished markdown report with
exactly these sections: ## Overview, ## Risk Analysis, ## Market Context,
## Portfolio Recommendations, ## Next Steps. Do not invent numbers that
weren't given to you. Keep it well-organized and skimmable.
"#;

/// Orchestrates the multi-agent portfolio analysis pipeline.
///
/// # Rust Concept: Composition over inheritance
/// `PortfolioOrchestrator` doesn't extend `ResearchAgent` — it owns its own
/// copies of the tools it needs and builds its own short-lived agents per
/// stage. This avoids coupling the two call patterns (single free-form
/// agent vs. a fixed pipeline) to the same struct.
pub struct PortfolioOrchestrator {
    config: Config,
    multi_tool: MultiSymbolTool,
    search_tool: WebSearchTool,
}

impl PortfolioOrchestrator {
    pub fn new(config: Config) -> Self {
        // 6 months of daily data gives a reasonable balance between enough
        // observations for correlation and not paying for years of history
        // the user didn't ask about.
        let multi_tool = MultiSymbolTool::new("6mo", "1d");
        // let search_tool = WebSearchTool::new(config.max_search_results);
        let search_tool = WebSearchTool::new(&config.tavily_api_key, config.max_search_results);
        Self {
            config,
            multi_tool,
            search_tool,
        }
    }

    fn ollama_client(&self) -> ollama::Client {
        std::env::set_var("OLLAMA_API_BASE_URL", &self.config.ollama_host);
        ollama::Client::from_env()
    }

    /// Run the full 5-stage pipeline for a basket of symbols.
    ///
    /// `context_query` is an optional free-text hint for the research
    /// stage (e.g. "emerging market tech exposure risks"). If `None`, a
    /// generic "recent news for these symbols" query is used instead.
    pub async fn run(&self, symbols: &[String], context_query: Option<&str>) -> Result<String> {
        info!(symbols = ?symbols, "Starting portfolio orchestration pipeline");

        // --- Stage 1: deterministic data gather + risk math (no LLM) ---
        let series = self.multi_tool.fetch_basket(symbols, None, None).await;
        let risk_summary = portfolio_analytics::compute_risk_summary(&series);

        let risk_markdown = match &risk_summary {
            Some(r) => r.to_markdown(),
            None => "_Not enough overlapping trading days across these symbols to compute \
                      correlation/portfolio risk (this can happen with very different exchange \
                      calendars or too few symbols). Proceeding with the data that was available._"
                .to_string(),
        };

        // --- Stage 2: quant agent interprets the numbers ---
        info!("Stage 2/5: quant agent");
        let quant_commentary = self.run_quant_agent(&risk_markdown).await?;

        // --- Stage 3: research agent gathers qualitative context ---
        info!("Stage 3/5: research agent");
        let research_query = context_query
            .map(|q| q.to_string())
            .unwrap_or_else(|| format!("Recent news and outlook for: {}", symbols.join(", ")));
        let research_context = self.run_research_agent(&research_query).await?;

        // --- Stage 4: portfolio agent synthesizes advice ---
        info!("Stage 4/5: portfolio agent");
        let portfolio_advice = self
            .run_portfolio_agent(&quant_commentary, &research_context)
            .await?;

        // --- Stage 5: report agent writes the final document ---
        info!("Stage 5/5: report agent");
        let report = self
            .run_report_agent(
                symbols,
                &risk_markdown,
                &quant_commentary,
                &research_context,
                &portfolio_advice,
            )
            .await?;

        info!("Portfolio orchestration pipeline completed");
        Ok(report)
    }

    async fn run_quant_agent(&self, risk_markdown: &str) -> Result<String> {
        let client = self.ollama_client();
        let agent = client
            .agent(&self.config.model)
            .preamble(QUANT_AGENT_PREAMBLE)
            .build();

        let prompt = format!("Here are the computed portfolio risk metrics:\n\n{}", risk_markdown);

        agent
            .prompt(&prompt)
            .await
            .map_err(|e| anyhow::anyhow!("Quant agent failed: {}", e))
    }

    async fn run_research_agent(&self, query: &str) -> Result<String> {
        let client = self.ollama_client();
        let agent = client
            .agent(&self.config.model)
            .preamble(RESEARCH_AGENT_PREAMBLE)
            .tool(self.search_tool.clone())
            .build();

        agent
            .prompt(query)
            .multi_turn(3)
            .await
            .map_err(|e| anyhow::anyhow!("Research agent failed: {}", e))
    }

    async fn run_portfolio_agent(&self, quant_commentary: &str, research_context: &str) -> Result<String> {
        let client = self.ollama_client();
        let agent = client
            .agent(&self.config.model)
            .preamble(PORTFOLIO_AGENT_PREAMBLE)
            .build();

        let prompt = format!(
            "Quant analyst's risk interpretation:\n{}\n\nMarket research context:\n{}",
            quant_commentary, research_context
        );

        agent
            .prompt(&prompt)
            .await
            .map_err(|e| anyhow::anyhow!("Portfolio agent failed: {}", e))
    }

    async fn run_report_agent(
        &self,
        symbols: &[String],
        risk_markdown: &str,
        quant_commentary: &str,
        research_context: &str,
        portfolio_advice: &str,
    ) -> Result<String> {
        let client = self.ollama_client();
        let agent = client
            .agent(&self.config.model)
            .preamble(REPORT_AGENT_PREAMBLE)
            .build();

        let prompt = format!(
            "Symbols: {}\n\nRaw risk metrics:\n{}\n\nQuant interpretation:\n{}\n\nMarket research:\n{}\n\nPortfolio strategy notes:\n{}",
            symbols.join(", "),
            risk_markdown,
            quant_commentary,
            research_context,
            portfolio_advice
        );

        agent
            .prompt(&prompt)
            .await
            .map_err(|e| anyhow::anyhow!("Report agent failed: {}", e))
    }
}

// =============================================================================
// UNIT TESTS
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orchestrator_creation() {
        let config = Config::default();
        let orchestrator = PortfolioOrchestrator::new(config);
        assert_eq!(orchestrator.config.model, "llama3.2");
    }

    #[test]
    fn test_preambles_are_not_empty() {
        assert!(!QUANT_AGENT_PREAMBLE.is_empty());
        assert!(!RESEARCH_AGENT_PREAMBLE.is_empty());
        assert!(!PORTFOLIO_AGENT_PREAMBLE.is_empty());
        assert!(!REPORT_AGENT_PREAMBLE.is_empty());
    }
}
