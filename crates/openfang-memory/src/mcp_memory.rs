//! Generic MCP memory service backend.
//!
//! OpenFang supports attaching any number of external memory services that speak
//! the [Model Context Protocol](https://modelcontextprotocol.io) over HTTP. Each
//! service exposes a `query` tool and is assigned a **rank** (lower = higher
//! priority). The recall chain tries services in rank order and stops at the first
//! service that responds — falling back to the next rank only when a service is
//! unreachable or times out. The internal SQLite store sits at a configurable rank
//! (`sqlite_rank`, default `1000`) so external services take precedence by default.
//!
//! ## Expected MCP tool contract
//!
//! The external service must expose a tool named `query` that accepts:
//! ```json
//! { "query": "<text>", "limit": <integer> }
//! ```
//! and returns an MCP text content block whose `text` field is a JSON array of
//! objects, each with at least `content: string` (required) and optionally
//! `score: f64` (0–1) and `path: string`.
//!
//! Example-compatible servers: [qmd](https://github.com/tobi/qmd), any custom
//! server built to this contract.

use openfang_types::{
    agent::AgentId,
    config::McpMemoryServiceConfig,
    memory::{MemoryFragment, MemoryId, MemorySource},
};
use std::collections::HashMap;

/// A live backend instance built from a [`McpMemoryServiceConfig`].
///
/// One instance is created per configured service at kernel boot and reused
/// for the lifetime of the daemon.
#[derive(Clone)]
pub struct McpMemoryBackend {
    client: reqwest::Client,
    /// Human-readable name for log messages.
    pub name: String,
    /// Rank in the fallback chain.
    pub rank: u32,
    /// Full MCP endpoint URL (`base_url + "/mcp"`).
    endpoint: String,
}

impl McpMemoryBackend {
    /// Create a backend from its config, building a dedicated `reqwest::Client`
    /// with the configured timeout baked in.
    pub fn new(cfg: &McpMemoryServiceConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(cfg.timeout_ms))
            .build()
            .expect("reqwest client build failed");
        Self {
            client,
            name: cfg.name.clone(),
            rank: cfg.rank,
            endpoint: format!("{}/mcp", cfg.mcp_url.trim_end_matches('/')),
        }
    }

    /// Query the remote MCP memory service.
    ///
    /// Returns:
    /// - `None` — service unreachable or timed out (caller should try next rank)
    /// - `Some(vec)` — service responded; vec may be empty (caller should stop
    ///   the chain and trust the service's answer)
    pub async fn query(&self, query: &str, limit: usize) -> Option<Vec<MemoryFragment>> {
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "query",
                "arguments": { "query": query, "limit": limit }
            }
        });

        let resp = match self.client.post(&self.endpoint).json(&body).send().await {
            Ok(r) => r,
            Err(e) => {
                tracing::debug!(
                    service = %self.name,
                    rank = self.rank,
                    error = %e,
                    "MCP memory service unreachable, trying next rank"
                );
                return None; // signal: try next rank
            }
        };

        let json: serde_json::Value = match resp.json().await {
            Ok(j) => j,
            Err(e) => {
                tracing::warn!(
                    service = %self.name,
                    "MCP memory service returned unparseable response: {e}"
                );
                return None; // treat parse failure as unavailable
            }
        };

        // Standard MCP text-content response:
        // { result: { content: [{ type: "text", text: "[{score?, path?, content}...]" }] } }
        let text = json
            .pointer("/result/content/0/text")
            .and_then(|v| v.as_str())
            .unwrap_or("[]");

        let entries: Vec<serde_json::Value> = serde_json::from_str(text).unwrap_or_default();

        let fragments = entries
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
                    "mcp_service".to_string(),
                    serde_json::Value::String(self.name.clone()),
                );

                Some(MemoryFragment {
                    id: MemoryId::new(),
                    // External MCP services are not scoped to a specific agent;
                    // they expose global/shared knowledge bases.
                    agent_id: AgentId(uuid::Uuid::nil()),
                    content,
                    embedding: None,
                    metadata,
                    source: MemorySource::Document,
                    confidence: score.clamp(0.0, 1.0),
                    created_at: chrono::Utc::now(),
                    accessed_at: chrono::Utc::now(),
                    access_count: 0,
                    scope: "mcp".to_string(),
                })
            })
            .collect();

        Some(fragments) // Some(empty) = service up but no results
    }
}
