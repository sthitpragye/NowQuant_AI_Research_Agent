//! # Agent Module
//!
//! This module implements the research agent using the Rig framework.
//! It demonstrates:
//! - Rig's agent builder pattern
//! - Tool integration for agentic workflows
//! - Async programming with tokio
//! - The Agent pattern in AI applications

use anyhow::Result;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::ollama;
use tracing::{debug, info};

use crate::config::Config;
use crate::tools::{FxRateTool, MultiSymbolTool, WebSearchTool, YahooFinanceTool, YahooFundamentalsTool};

// =============================================================================
// SYSTEM PROMPT
// =============================================================================
/// The system prompt defines the agent's personality and behavior.
const RESEARCH_SYSTEM_PROMPT: &str = r#"
You are a financial analyst assistant with access to real-time market data tools.

TOOLS AVAILABLE:
- yahoo_finance: single stock price, candles, quant stats
- yahoo_fundamentals: P/E, market cap, margins, valuation
- web_search: news and qualitative context
- multi_symbol_snapshot: ONLY use when the user asks about MULTIPLE tickers
- fx_rate: ONLY use when the user explicitly asks about exchange rates

STRICT RULES:
1. For a single stock query, ONLY call yahoo_finance and yahoo_fundamentals. Never call multi_symbol_snapshot or fx_rate.
2. multi_symbol_snapshot requires symbols as a JSON array: ["AAPL", "MSFT"] — never as a string.
3. Never suggest or write Python code. You have tools — use them.
4. Never say a tool failed if it returned data. Synthesise what you received.
5. After tools return data, write your analysis immediately. Do not call more tools.
6. CRITICAL: Always use the exact numbers returned by tools. Never substitute, estimate, round, or recompute price figures unless explicitly requested.
7. For single stock analysis, always call yahoo_finance with range=3mo and interval=1d unless the user explicitly specifies different values.
8. If yahoo_finance returns a price or performance metric, report those values exactly as returned by the tool.

OUTPUT FORMAT:
## [TICKER] — Analysis
**Price & Performance** (from yahoo_finance)
**Fundamentals** (from yahoo_fundamentals)
**Market Context** (from web_search)
**Assessment** — 4-5 sentences synthesising all three sources
"#;

// =============================================================================
// RESEARCH AGENT STRUCT
// =============================================================================
/// The main research agent that orchestrates LLM + tools.
///
/// # Rust Concept: Struct with References
///
/// We store a Config by value (owned). This means ResearchAgent owns
/// its configuration and will clean it up when dropped.
pub struct ResearchAgent {
    /// Configuration for the agent
    config: Config,

    /// The web search tool
    search_tool: WebSearchTool,

    /// The Yahoo Finance price/history scraping tool
    finance_tool: YahooFinanceTool,

    /// The Yahoo Finance fundamentals (valuation/margins) scraping tool
    fundamentals_tool: YahooFundamentalsTool,

    /// The FX rate tool (currency pairs)
    fx_tool: FxRateTool,

    /// The multi-symbol (portfolio basket) snapshot tool
    multi_tool: MultiSymbolTool,
}

impl ResearchAgent {
    /// Create a new ResearchAgent with the given configuration.
    ///
    /// # Rust Concept: Constructor Pattern
    ///
    /// Rust doesn't have constructors like OOP languages.
    /// Instead, we use associated functions (usually named `new`).
    pub fn new(config: Config) -> Self {
        // let search_tool = WebSearchTool::new(config.max_search_results);
        let search_tool = WebSearchTool::new(&config.tavily_api_key, config.max_search_results);
        let finance_tool = YahooFinanceTool::new("3mo", "1d");
        let fundamentals_tool = YahooFundamentalsTool::new();
        let fx_tool = FxRateTool::new();
        let multi_tool = MultiSymbolTool::new("3mo", "1d");

        Self {
            config,
            search_tool,
            finance_tool,
            fundamentals_tool,
            fx_tool,
            multi_tool,
        }
    }

    /// Research a topic and return a comprehensive summary.
    ///
    /// # Rust Concept: Ownership and Borrowing
    ///
    /// `&self` means we borrow the ResearchAgent immutably.
    /// `&str` for the query borrows the string data without copying.
    pub async fn research(&self, query: &str) -> Result<String> {
        info!(query = %query, "Starting research task");

        // Step 1: Create the Ollama client using the builder pattern
        // In Rig 0.27, use ollama::Client::from_env() which reads OLLAMA_API_BASE_URL
        // environment variable, or defaults to http://localhost:11434
        //
        // # Rust Concept: Environment Variable Configuration
        // Instead of hardcoding values, we use environment variables.
        // This is a 12-factor app best practice for configuration.
        std::env::set_var("OLLAMA_API_BASE_URL", &self.config.ollama_host);

        let ollama_client = ollama::Client::from_env();

        debug!(
            host = %self.config.ollama_host,
            model = %self.config.model,
            "Connected to Ollama"
        );

        // Step 2: Build the agent with tools
        //
        // Rig's agent builder lets us:
        // - Set the model
        // - Add a system prompt (preamble)
        // - Register tools the agent can use
        let agent = ollama_client
            .agent(&self.config.model)
            .preamble(RESEARCH_SYSTEM_PROMPT)
            .tool(self.search_tool.clone())
            .tool(self.finance_tool.clone())
            .tool(self.fundamentals_tool.clone())
            .tool(self.fx_tool.clone())
            .tool(self.multi_tool.clone())
            .build();

        info!("Agent configured, executing research query");

        // Step 3: Execute the research query
        let enhanced_query = format!(
            "Research the following request.

        If it is a SINGLE STOCK:
        - Call yahoo_finance FIRST using range=3mo and interval=1d unless the user specifies otherwise.
        - Then call yahoo_fundamentals.
        - Use web_search only for qualitative news/context.
        - Use the exact numeric values returned by the tools. Never substitute or invent prices.

        If it is MULTIPLE STOCKS:
        - Use multi_symbol_snapshot.

        Then write your analysis immediately after the tool results.

        User request:
        {}",
            query
        );

        let response = agent
            .prompt(&enhanced_query)
            .multi_turn(5) // Allow up to 5 iterations of tool calls
            .await
            .map_err(|e| anyhow::anyhow!("Agent execution failed: {}", e))?;

        info!("Research completed successfully");

        Ok(response)
    }

    /// Perform a quick search without full agent reasoning.
    ///
    /// This is useful when you just want search results without
    /// the agent synthesizing them.
    pub async fn quick_search(&self, query: &str) -> Result<String> {
        info!(query = %query, "Performing quick search");

        let results = self
            .search_tool
            .search(query)
            .await
            .map_err(|e| anyhow::anyhow!("Search failed: {}", e))?;

        if results.is_empty() {
            return Ok(format!("No results found for: {}", query));
        }

        // Format results nicely
        let formatted: String = results
            .iter()
            .enumerate()
            .map(|(i, r)| {
                format!(
                    "{}. **{}**\n   {}\n   URL: {}\n",
                    i + 1,
                    r.title,
                    r.content,
                    r.url
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        Ok(format!("## Search Results\n\n{}", formatted))
    }

    /// Perform a direct Yahoo Finance lookup without LLM reasoning.
    ///
    /// This is the financial equivalent of `quick_search`: it calls the
    /// `yahoo_finance` tool directly and returns its formatted markdown
    /// report (quote snapshot + computed quant stats + recent candles)
    /// without spending any LLM tokens. Useful for fast, deterministic
    /// data pulls (e.g. scripted portfolio checks) or when Ollama isn't
    /// running.
    pub async fn analyze_symbol(&self, symbol: &str) -> Result<String> {
        info!(symbol = %symbol, "Performing direct Yahoo Finance lookup");

        self.finance_tool
            .get_report(symbol, None, None)
            .await
            .map_err(|e| anyhow::anyhow!("Yahoo Finance lookup failed: {}", e))
    }

    /// Direct fundamentals lookup without LLM reasoning.
    pub async fn get_fundamentals(&self, symbol: &str) -> Result<String> {
        info!(symbol = %symbol, "Performing direct Yahoo Finance fundamentals lookup");

        self.fundamentals_tool
            .get_report(symbol)
            .await
            .map_err(|e| anyhow::anyhow!("Yahoo Finance fundamentals lookup failed: {}", e))
    }

    /// Direct FX rate lookup without LLM reasoning.
    pub async fn get_fx_rate(&self, from_currency: &str, to_currency: &str) -> Result<String> {
        info!(from = %from_currency, to = %to_currency, "Performing direct FX rate lookup");

        self.fx_tool
            .get_report(from_currency, to_currency)
            .await
            .map_err(|e| anyhow::anyhow!("FX rate lookup failed: {}", e))
    }

    /// Direct multi-symbol portfolio snapshot without LLM reasoning.
    pub async fn get_portfolio_snapshot(&self, symbols: &[String]) -> Result<String> {
        info!(symbols = ?symbols, "Performing direct multi-symbol snapshot lookup");

        Ok(self.multi_tool.get_report(symbols, None, None).await)
    }
}

// =============================================================================
// UNIT TESTS
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_creation() {
        let config = Config::default();
        let agent = ResearchAgent::new(config);

        assert_eq!(agent.config.model, "llama3.2");
    }

    #[test]
    fn test_system_prompt_not_empty() {
        assert!(!RESEARCH_SYSTEM_PROMPT.is_empty());
        assert!(RESEARCH_SYSTEM_PROMPT.contains("research"));
    }
}
