//! # AI Research Agent
//! 
//! A production-ready AI research agent built with the Rig framework.
//! 
//! This application demonstrates:
//! - Building AI agents in Rust
//! - Using Ollama for local LLM inference
//! - Web search integration with DuckDuckGo
//! - CLI design with clap
//! - Structured logging with tracing
//! - Error handling best practices
//! 
//! ## Quick Start
//! ```bash
//! cargo run -- "What are the latest developments in Rust?"
//! ```

// =============================================================================
// MODULE DECLARATIONS
// =============================================================================
// Rust requires explicit module declarations. Each `mod` statement tells
// the compiler to look for a file with that name (e.g., config.rs).

/// Configuration management
mod config;

/// Research agent implementation
mod agent;

/// Web search and other tools
mod tools;

/// Pure-Rust portfolio risk math (correlation, volatility, diversification)
mod portfolio_analytics;

/// Multi-agent portfolio analysis pipeline
mod orchestrator;

// =============================================================================
// IMPORTS
// =============================================================================
use anyhow::Result;
use clap::Parser;
use tracing::{error, info, Level};
use tracing_subscriber::FmtSubscriber;

use crate::agent::ResearchAgent;
use crate::config::Config;

// =============================================================================
// CLI ARGUMENTS
// =============================================================================
/// # Rust Concept: Derive Macros with Clap
/// 
/// Clap's derive feature lets us define CLI arguments as a struct.
/// The macros automatically generate argument parsing code.
/// 
/// - #[command(...)]: Configures the overall program
/// - #[arg(...)]: Configures individual arguments
#[derive(Parser, Debug)]
#[command(
    name = "ai-research-agent",
    author = "Your Name",
    version = "0.1.0",
    about = "An AI-powered research assistant that searches the web and summarizes findings",
    long_about = r#"
AI Research Agent - Your intelligent research companion!

This tool uses local LLMs (via Ollama) and web search to help you research any topic.
It will:
  1. Search the web for relevant information
  2. Analyze and synthesize the results
  3. Provide a comprehensive summary with sources

PREREQUISITES:
  1. Install Ollama: https://ollama.ai
  2. Pull a model: ollama pull llama3.2
  3. Start Ollama: ollama serve

EXAMPLES:
  # Basic research query
  ai-research-agent "What are the latest developments in Rust async?"
  
  # Quick search without synthesis
  ai-research-agent --quick "Rust web frameworks 2024"

  # Direct Yahoo Finance lookups (no LLM needed)
  ai-research-agent --finance "AAPL"
  ai-research-agent --finance "RELIANCE.NS"
  ai-research-agent --fundamentals "PETR4.SA"
  ai-research-agent --fx "USD,INR"
  ai-research-agent --portfolio "AAPL,MSFT,RELIANCE.NS,PETR4.SA"

  # Multi-agent portfolio analysis pipeline (quant + research + portfolio + report agents)
  ai-research-agent --orchestrate "AAPL,MSFT,RELIANCE.NS,PETR4.SA"
  ai-research-agent --orchestrate "AAPL,RELIANCE.NS" --context "AI capex exposure"
  
  # Use a specific model
  ai-research-agent --model deepseek-v3.2 "Machine learning in Rust"
"#
)]
struct Args {
    /// The research topic or question to investigate
    #[arg(
        help = "The topic to research",
        value_name = "QUERY"
    )]
    query: String,
    
    /// The Ollama model to use (overrides OLLAMA_MODEL env var)
    #[arg(
        short = 'm',
        long = "model",
        help = "Ollama model to use",
        env = "OLLAMA_MODEL"
    )]
    model: Option<String>,
    
    /// Quick search mode - just search, don't synthesize
    #[arg(
        short = 'q',
        long = "quick",
        help = "Quick search mode (no AI synthesis)",
        default_value = "false"
    )]
    quick: bool,

    /// Direct Yahoo Finance lookup mode - treats QUERY as a ticker symbol
    #[arg(
        short = 'f',
        long = "finance",
        help = "Direct Yahoo Finance price/history lookup (QUERY = ticker symbol, e.g. AAPL or RELIANCE.NS, no AI synthesis)",
        default_value = "false"
    )]
    finance: bool,

    /// Direct Yahoo Finance fundamentals lookup mode - treats QUERY as a ticker symbol
    #[arg(
        long = "fundamentals",
        help = "Direct Yahoo Finance fundamentals lookup (QUERY = ticker symbol, no AI synthesis)",
        default_value = "false"
    )]
    fundamentals: bool,

    /// Direct FX rate lookup mode - treats QUERY as "FROM,TO" e.g. "USD,INR"
    #[arg(
        long = "fx",
        help = "Direct FX rate lookup (QUERY = \"FROM,TO\" currency codes, e.g. \"USD,INR\", no AI synthesis)",
        default_value = "false"
    )]
    fx: bool,

    /// Direct portfolio snapshot mode - treats QUERY as comma-separated ticker symbols
    #[arg(
        long = "portfolio",
        help = "Direct multi-symbol portfolio snapshot (QUERY = comma-separated tickers, e.g. \"AAPL,MSFT,RELIANCE.NS\", no AI synthesis)",
        default_value = "false"
    )]
    portfolio: bool,

    /// Multi-agent portfolio analysis pipeline - treats QUERY as comma-separated ticker symbols
    #[arg(
        long = "orchestrate",
        help = "Run the multi-agent portfolio analysis pipeline (QUERY = comma-separated tickers, e.g. \"AAPL,MSFT,RELIANCE.NS\"). Needs Ollama running.",
        default_value = "false"
    )]
    orchestrate: bool,

    /// Optional free-text research context for --orchestrate's research stage
    #[arg(
        long = "context",
        help = "Optional research focus for --orchestrate (e.g. \"emerging market tech exposure risks\"); defaults to a generic news lookup for the given symbols"
    )]
    context: Option<String>,
    
    /// Verbose output (debug logging)
    #[arg(
        short = 'v',
        long = "verbose",
        help = "Enable verbose/debug logging",
        default_value = "false"
    )]
    verbose: bool,
}

// =============================================================================
// MAIN FUNCTION
// =============================================================================
/// # Rust Concept: The #[tokio::main] Attribute
/// 
/// Rust's main() function is synchronous by default.
/// #[tokio::main] transforms it into an async function by:
/// 1. Creating a Tokio runtime
/// 2. Running our async main inside it
/// 
/// This is equivalent to:
/// ```
/// fn main() {
///     let rt = tokio::runtime::Runtime::new().unwrap();
///     rt.block_on(async { /* our code */ });
/// }
/// ```
#[tokio::main]
async fn main() -> Result<()> {
    // Parse command-line arguments
    // Clap handles --help, --version, and error messages automatically
    let args = Args::parse();
    
    // Initialize logging
    init_logging(args.verbose)?;
    
    info!("AI Research Agent starting up...");
    
    // Load configuration from environment/.env file
    let mut config = Config::from_env()?;
    
    // Override model if specified on command line
    // 
    // # Rust Concept: Option Type
    // Option<T> is either Some(value) or None.
    // if let Some(x) = option { } is a concise way to handle this.
    if let Some(model) = args.model {
        info!(model = %model, "Using model from command line");
        config.model = model;
    }
    
    // Validate configuration
    config.validate()?;
    
    info!(
        model = %config.model,
        host = %config.ollama_host,
        "Configuration loaded"
    );
    
    // Create the research agent (clone Config so it's still available below
    // for the orchestrator path, which owns its own tool instances).
    let agent = ResearchAgent::new(config.clone());

    // Execute the query
    let result = if args.orchestrate {
        info!("Running multi-agent portfolio orchestration pipeline");
        let orchestrator = orchestrator::PortfolioOrchestrator::new(config);
        let symbols: Vec<String> = args.query.split(',').map(|s| s.trim().to_string()).collect();
        orchestrator.run(&symbols, args.context.as_deref()).await
    } else if args.fundamentals {
        info!("Running in direct Yahoo Finance fundamentals lookup mode");
        agent.get_fundamentals(&args.query).await
    } else if args.fx {
        info!("Running in direct FX rate lookup mode");
        let parts: Vec<&str> = args.query.split(',').map(|s| s.trim()).collect();
        match parts.as_slice() {
            [from, to] => agent.get_fx_rate(from, to).await,
            _ => Err(anyhow::anyhow!(
                "--fx expects QUERY in the form \"FROM,TO\", e.g. \"USD,INR\""
            )),
        }
    } else if args.portfolio {
        info!("Running in direct portfolio snapshot mode");
        let symbols: Vec<String> = args.query.split(',').map(|s| s.trim().to_string()).collect();
        agent.get_portfolio_snapshot(&symbols).await
    } else if args.finance {
        // Finance mode: direct Yahoo Finance lookup, no LLM
        info!("Running in direct Yahoo Finance lookup mode");
        agent.analyze_symbol(&args.query).await
    } else if args.quick {
        // Quick mode: just search, no synthesis
        info!("Running in quick search mode");
        agent.quick_search(&args.query).await
    } else {
        // Full mode: search + AI synthesis
        info!("Running full research mode");
        agent.research(&args.query).await
    };
    
    // Handle the result
    match result {
        Ok(response) => {
            // Print the result to stdout
            println!("\n{}", "=".repeat(60));
            println!("RESEARCH RESULTS");
            println!("{}\n", "=".repeat(60));
            println!("{}", response);
            println!("\n{}", "=".repeat(60));
        }
        Err(e) => {
            // Print a user-friendly error message
            error!(error = %e, "Research failed");
            
            // Give helpful suggestions based on common errors
            eprintln!("\n❌ Research failed: {}", e);
            
            if e.to_string().contains("connection refused") {
                eprintln!("\n💡 Tip: Make sure Ollama is running:");
                eprintln!("   ollama serve");
            } else if e.to_string().contains("model") {
                eprintln!("\n💡 Tip: Make sure the model is installed:");
                eprintln!("   ollama pull llama3.2");
            }
            
            // Return the error to set non-zero exit code
            return Err(e);
        }
    }
    
    info!("Research completed successfully");
    Ok(())
}

// =============================================================================
// LOGGING INITIALIZATION
// =============================================================================
/// Initialize the tracing subscriber for structured logging.
/// 
/// # Rust Concept: Early Returns
/// 
/// The `?` operator returns early from the function if there's an error.
/// This is common in initialization code where failure should abort.
fn init_logging(verbose: bool) -> Result<()> {
    // Set log level based on verbose flag
    let level = if verbose { Level::DEBUG } else { Level::INFO };
    
    // Build the subscriber
    // 
    // # Rust Concept: Builder Pattern
    // Many Rust libraries use builders for configuration.
    // Each method modifies the builder and returns it for chaining.
    let subscriber = FmtSubscriber::builder()
        .with_max_level(level)
        .with_target(true)  // Show the module that logged
        .with_thread_names(false)
        .with_file(false)
        .with_line_number(false)
        .finish();
    
    // Set as the global default
    tracing::subscriber::set_global_default(subscriber)
        .map_err(|e| anyhow::anyhow!("Failed to set logging subscriber: {}", e))?;
    
    Ok(())
}

// =============================================================================
// INTEGRATION TESTS
// =============================================================================
/// # Rust Concept: Integration Tests
/// 
/// These tests check that all components work together.
/// They're placed in the same module but could also be in tests/ directory.
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_args_parsing() {
        // Test that CLI args parse correctly
        let args = Args::parse_from(["test", "What is Rust?"]);
        assert_eq!(args.query, "What is Rust?");
        assert!(!args.quick);
        assert!(!args.verbose);
    }
    
    #[test]
    fn test_args_with_flags() {
        let args = Args::parse_from([
            "test",
            "--quick",
            "--verbose",
            "--model", "llama3.2",
            "Test query"
        ]);
        
        assert_eq!(args.query, "Test query");
        assert!(args.quick);
        assert!(args.verbose);
        assert_eq!(args.model, Some("llama3.2".to_string()));
    }
}
