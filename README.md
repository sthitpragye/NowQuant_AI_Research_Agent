# Quant Research Agent — A Rust-Native Financial Intelligence Framework

A modular, Rust-based multi-agent framework that pairs **local LLM reasoning** (via Ollama) with **structured, deterministic financial data tools** to automate quantitative research, strategy refinement, and portfolio risk analysis — with explicit support for emerging-market tickers (NSE/India, B3/Brazil, and other Yahoo Finance-listed exchanges) alongside developed markets.

Built on the [Rig](https://rig.rs) agent framework, this project explores a core design question for financial AI systems: **how much of the pipeline should the LLM actually touch?** The answer here is "as little as possible" — all price math, correlation, volatility, and diversification metrics are computed in pure Rust; the LLM is only ever asked to *interpret* numbers that are already known to be correct, never to calculate them.

![Rust](https://img.shields.io/badge/Rust-1.70+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)
![LLM](https://img.shields.io/badge/LLM-Ollama%20(local)-green.svg)
![Markets](https://img.shields.io/badge/Markets-Global%20%2B%20Emerging-purple.svg)

---

## Why this project exists

Most "AI finance agent" demos either (a) let an LLM freely call tools and hope it reports numbers faithfully, or (b) hardcode a single rigid pipeline with no LLM reasoning at all. This project sits deliberately between the two, and treats that boundary as the main engineering problem:

- **Structured tool abstractions** wrap noisy, heterogeneous market data (Yahoo Finance chart/quote endpoints, FX pairs, fundamentals) into typed Rust structs the LLM can consume but never has to parse or recompute.
- **A strict separation between computation and interpretation** — correlation matrices, annualized volatility, drawdown, and diversification ratios are computed once in `portfolio_analytics.rs` and handed to the LLM as ground truth. The system prompts explicitly forbid the model from recalculating, rounding, or substituting any returned figure.
- **A fixed, auditable multi-agent pipeline** instead of a free-form planner — because for financial reporting, predictability and reproducibility matter more than autonomy. See [Feasibility notes](#feasibility-notes-on-agent-frameworks) below for why this tradeoff was made deliberately.
- **First-class emerging-market support** — ticker suffixes like `.NS` (NSE, India), `.SA` (B3, Brazil), `.L` (LSE), and similar are first-class citizens throughout, not an afterthought bolted onto a US-equities tool.

---

## Architecture

```
                         ┌────────────────────────┐
                         │   CLI (clap) — main.rs  │
                         └────────────┬─────────────┘
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            │                         │                         │
   ┌────────▼────────┐      ┌─────────▼─────────┐      ┌─────────▼─────────┐
   │  ResearchAgent   │      │ PortfolioOrches-  │      │  Direct tool calls │
   │  (single agent,  │      │ trator (5-stage   │      │  (--finance, --fx, │
   │  free-form +     │      │ fixed LLM         │      │  --portfolio, no   │
   │  multi-turn tool  │      │ pipeline)          │      │  LLM, deterministic)│
   │  use, Rig)       │      │                    │      │                    │
   └────────┬────────┘      └─────────┬─────────┘      └─────────┬─────────┘
            │                         │                          │
            └─────────────┬───────────┴──────────────────────────┘
                          │
              ┌────────────▼─────────────┐
              │   tools.rs (typed tools)  │
              │  yahoo_finance            │
              │  yahoo_fundamentals       │
              │  fx_rate                  │
              │  multi_symbol_snapshot    │
              │  web_search (Tavily)      │
              └────────────┬─────────────┘
                           │
              ┌────────────▼─────────────┐
              │ portfolio_analytics.rs    │
              │ (pure Rust: correlation,  │
              │ volatility, diversification│
              │ ratio — never touches LLM) │
              └───────────────────────────┘
```

---

## Features

- **Local-first LLM inference** via Ollama — no API key, no data leaving your machine for the reasoning layer.
- **Deterministic quant core** — Pearson correlation, annualized volatility, max drawdown, and a diversification ratio computed in plain Rust floating-point math, unit-tested for correctness (identical-series, single-symbol, and partial-failure cases).
- **Five structured financial tools**, each a typed Rig `Tool` implementation:
  | Tool | Purpose |
  |---|---|
  | `yahoo_finance` | Single-symbol price history, candles, and derived quant stats (volatility, return, drawdown, 52w range) |
  | `yahoo_fundamentals` | P/E, market cap, margins, valuation metrics for a symbol |
  | `fx_rate` | Spot exchange rate lookup between two currency codes |
  | `multi_symbol_snapshot` | Basket-level fetch across multiple tickers in one call (feeds the risk engine) |
  | `web_search` | Tavily-backed qualitative news/context search |
- **Global + emerging market coverage** — works with any Yahoo Finance-resolvable ticker, including NSE (`RELIANCE.NS`), B3 (`PETR4.SA`), LSE, and other non-US exchanges, with currency/exchange metadata surfaced explicitly in every report.
- **Multi-agent orchestration pipeline** (`--orchestrate`) — five narrowly-scoped LLM stages (quant interpretation → market research → portfolio strategy → report writer) chained over deterministically-computed data.
- **Deterministic fast paths** — `--finance`, `--fundamentals`, `--fx`, and `--portfolio` hit the tools directly with zero LLM calls, for scripted/repeatable data pulls.
- **Structured report generation** — both the single-agent and orchestrator paths are constrained by system prompts to emit consistent markdown sections (Overview, Risk Analysis, Market Context, Recommendations, Next Steps).
- **Unit + integration tested** — config validation, agent construction, preamble sanity checks, and the quant math itself all have test coverage.

---

## Quick start

### Prerequisites

```bash
# 1. Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 2. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 3. Pull a model
ollama pull llama3.2

# 4. Start Ollama
ollama serve
```

### Setup

```bash
git clone <this-repo>
cd quant-research-agent
cp .env.example .env   # add TAVILY_API_KEY for web_search
cargo build --release
```

### Usage

```bash
# Full agentic research (LLM + tools, multi-turn)
cargo run -- "Give me a quant and fundamentals view of AAPL"

# Direct, LLM-free Yahoo Finance lookup (developed market)
cargo run -- --finance "AAPL"

# Direct lookup — emerging market (India, NSE)
cargo run -- --finance "RELIANCE.NS"

# Fundamentals — emerging market (Brazil, B3)
cargo run -- --fundamentals "PETR4.SA"

# FX rate lookup
cargo run -- --fx "USD,INR"

# Multi-symbol portfolio snapshot across markets
cargo run -- --portfolio "AAPL,MSFT,RELIANCE.NS,PETR4.SA"

# Full 5-stage multi-agent portfolio analysis pipeline
cargo run -- --orchestrate "AAPL,MSFT,RELIANCE.NS,PETR4.SA"

# ...with a custom research focus for stage 3
cargo run -- --orchestrate "AAPL,RELIANCE.NS" --context "AI capex exposure in EM tech"

# Use a different model / verbose logging
cargo run -- --model deepseek-v3.2 --verbose "Emerging market equities outlook"
```

---

## The multi-agent orchestration pipeline (`--orchestrate`)

Rig — like most lightweight Rust agent frameworks — doesn't ship a built-in multi-agent graph/planner the way LangGraph or AutoGen do in the Python ecosystem. The orchestrator is the manual equivalent: a fixed, auditable sequence of specialized LLM calls, each with a narrow preamble and a single job.

| Stage | Type | Responsibility |
|---|---|---|
| 1. Data gather | Deterministic, no LLM | Fetch basket price history via `multi_symbol_snapshot` |
| 2. Quant interpretation | LLM, no tools | Read pre-computed correlation/volatility/diversification numbers and explain what they mean — never recompute them |
| 3. Market research | LLM + `web_search` | Pull qualitative news/context for the basket or a user-supplied focus |
| 4. Portfolio strategy | LLM, no tools | Combine quant interpretation + research into actionable observations |
| 5. Report writer | LLM, no tools | Produce one polished markdown report from everything above |

This costs some autonomy versus a planner-driven framework — there's no model deciding the stage order at runtime — but for financial reporting that's the point: every run takes the same shape, every number in the final report is traceable back to a specific deterministic computation, and any stage can be tested or swapped independently.

---

## Feasibility notes on agent frameworks

A short summary of the design tradeoffs explored while building this:

- **Free-form agent + tools (Rig's `multi_turn`)** — good for exploratory, single-symbol research where the question shape varies. Risk: an undisciplined agent can call tools redundantly or attempt arithmetic itself, which is why the system prompt enumerates strict rules (no recomputation, no invented numbers, no Python suggestions).
- **Fixed multi-agent pipeline (this orchestrator)** — better fit for repeatable, structured deliverables like portfolio reports, where reproducibility and number-traceability matter more than flexibility. This is the more "production-feasible" pattern for anything that resembles a real research note.
- **Heavier Python frameworks (LangGraph, AutoGen, CrewAI)** offer built-in planners and graph-based control flow that Rig does not. They were not adopted here because the explicit goal was a backend intelligence layer in a single, dependency-light Rust binary — the control flow is implemented by hand instead, trading some framework convenience for auditability and a smaller deployment surface.
- **Emerging markets** add real friction at the data layer, not the agent layer: different trading calendars (handled by intersecting aligned trading days before computing correlation — see `portfolio_analytics::compute_risk_summary`), different currencies (surfaced per-symbol rather than normalized, since silent FX conversion would hide real risk), and sparser/noisier fundamentals data from Yahoo Finance. The framework is built to degrade gracefully (excluding a bad symbol rather than failing the whole basket) rather than to silently paper over these gaps.

---

## Project structure

```
quant-research-agent/
├── Cargo.toml                 # Dependencies: rig-core, tokio, reqwest, clap, serde, tracing...
├── .env.example                # OLLAMA_MODEL, OLLAMA_HOST, TAVILY_API_KEY, TEMPERATURE, ...
├── README.md
└── src/
    ├── main.rs                # CLI surface (clap) — routes to agent / orchestrator / direct tool calls
    ├── config.rs               # Env-driven configuration + validation
    ├── agent.rs                 # Single-agent ResearchAgent: Rig builder, system prompt, tool registration
    ├── tools.rs                 # Typed Rig Tool implementations (Yahoo Finance, fundamentals, FX, basket, web search)
    ├── portfolio_analytics.rs  # Pure-Rust risk math: correlation matrix, volatility, diversification ratio
    └── orchestrator.rs          # 5-stage multi-agent portfolio pipeline
```

## Configuration (`.env`)

```bash
OLLAMA_MODEL=llama3.2          # Any Ollama-installed model
OLLAMA_HOST=http://localhost:11434
TEMPERATURE=0.7                 # 0.0 = focused, 2.0 = creative
MAX_SEARCH_RESULTS=5
TAVILY_API_KEY=                 # Required for the web_search tool
RUST_LOG=info
```

## Testing

```bash
cargo test                  # all unit + integration tests
cargo test -- --nocapture   # with output
cargo test test_config      # a specific test
```

Coverage includes: config validation boundaries, agent/orchestrator construction, system-prompt sanity checks, and the quant engine (identical-series correlation, single-symbol rejection, graceful exclusion of errored symbols from a basket).

## Roadmap / known limitations

- Currently single-process CLI; no persistent job queue or API server for scheduled/automated runs.
- Position sizing is equal-weighted only — no optimizer (mean-variance, risk parity, etc.) yet.
- Yahoo Finance scraping is unauthenticated and rate-limit sensitive; not a substitute for a licensed market data feed in production use.
- No backtesting module yet — the framework currently supports point-in-time analysis, not historical strategy evaluation.
- FX is surfaced per-symbol rather than netted into a base-currency portfolio view.

## License

MIT — see `LICENSE`.
