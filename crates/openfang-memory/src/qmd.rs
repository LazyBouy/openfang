//! QmdRecallBackend — HTTP client for the qmd MCP server.
//!
//! qmd (<https://github.com/tobi/qmd>) is a local hybrid BM25 + vector + LLM
//! re-ranker search engine for markdown knowledge bases. When a qmd MCP daemon
//! is running (`qmd mcp --http --daemon`), this backend queries it in parallel
//! with the local SQLite semantic store and merges the results into every agent
//! recall, transparently enriching context with knowledge from indexed documents.
//!
//! If the qmd daemon is unreachable the backend returns an empty result set and
//! logs a DEBUG message — the local recall path continues unaffected.

use openfang_types::{
    agent::AgentId,
    memory::{MemoryFragment, MemoryId, MemorySource},
};
use std::collections::HashMap;

/// Configuration for connecting to a running qmd MCP HTTP server.
#[derive(Debug, Clone)]
pub struct QmdConfig {
    /// Base URL of the qmd MCP daemon, e.g. `"http://127.0.0.1:7384"`.
    pub base_url: String,
    /// Per-request timeout in milliseconds.
    pub timeout_ms: u64,
}

impl Default for QmdConfig {
    fn default() -> Self {
        Self {
            base_url: "http://127.0.0.1:7384".to_string(),
            timeout_ms: 2000,
        }
    }
}

/// Async HTTP backend that queries a qmd MCP server for memory recall.
#[derive(Clone)]
pub struct QmdRecallBackend {
    client: reqwest::Client,
    config: QmdConfig,
}

impl QmdRecallBackend {
    /// Create a new backend with the given config.
    pub fn new(config: QmdConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(config.timeout_ms))
            .build()
            .expect("reqwest client build failed");
        Self { client, config }
    }

    /// Query the qmd index for `query`, returning up to `limit` memory fragments.
    ///
    /// Returns an empty `Vec` on any network error or timeout (graceful degradation).
    pub async fn query(&self, query: &str, limit: usize) -> Vec<MemoryFragment> {
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "query",
                "arguments": { "query": query, "limit": limit }
            }
        });

        let resp = match self
            .client
            .post(format!("{}/mcp", self.config.base_url))
            .json(&body)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::debug!("qmd MCP unreachable ({}), skipping qmd recall", e);
                return vec![];
            }
        };

        let json: serde_json::Value = match resp.json().await {
            Ok(j) => j,
            Err(e) => {
                tracing::warn!("qmd MCP response parse error: {}", e);
                return vec![];
            }
        };

        // qmd MCP response shape:
        // { result: { content: [{ type: "text", text: "[{score, path, content}, ...]" }] } }
        let text = json
            .pointer("/result/content/0/text")
            .and_then(|v| v.as_str())
            .unwrap_or("[]");

        let results: Vec<serde_json::Value> = serde_json::from_str(text).unwrap_or_default();

        results
            .into_iter()
            .filter_map(|r| {
                let content = r.get("content")?.as_str()?.to_string();
                let score = r
                    .get("score")
                    .and_then(|s| s.as_f64())
                    .unwrap_or(0.5) as f32;
                let path = r
                    .get("path")
                    .and_then(|p| p.as_str())
                    .unwrap_or("")
                    .to_string();

                let mut metadata = HashMap::new();
                metadata.insert(
                    "source_path".to_string(),
                    serde_json::Value::String(path),
                );
                metadata.insert(
                    "backend".to_string(),
                    serde_json::Value::String("qmd".to_string()),
                );

                Some(MemoryFragment {
                    id: MemoryId::new(),
                    // agent_id is a nil placeholder; the merge step in substrate.rs
                    // does not filter qmd results by agent_id (they are global documents).
                    agent_id: AgentId(uuid::Uuid::nil()),
                    content,
                    embedding: None,
                    metadata,
                    source: MemorySource::Document,
                    confidence: score.clamp(0.0, 1.0),
                    created_at: chrono::Utc::now(),
                    accessed_at: chrono::Utc::now(),
                    access_count: 0,
                    scope: "qmd".to_string(),
                })
            })
            .collect()
    }
}
