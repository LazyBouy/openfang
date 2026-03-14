//! QMD-compatible memory service backend.
//!
//! OpenFang connects to external memory services via a simple REST query API.
//! Each service is assigned a **rank** (lower = higher priority). The recall
//! chain tries services in rank order and stops at the first service that
//! responds — falling back to the next rank only when a service is unreachable,
//! times out, or returns an HTTP error status.
//!
//! The internal SQLite store sits at `sqlite_rank` (default `1000`) in the same
//! chain, so external services take precedence by default.
//!
//! ## REST query contract
//!
//! The external service must expose `POST /query` accepting:
//! ```json
//! {
//!   "searches": [{ "type": "vec", "query": "<text>" }],
//!   "limit": <integer>
//! }
//! ```
//! and respond with:
//! ```json
//! {
//!   "results": [
//!     { "file": "path/to/doc.md", "title": "...", "snippet": "...", "score": 0.85 }
//!   ]
//! }
//! ```
//! `score` (0–1) and `file` are optional. `title` and/or `snippet` must be
//! present — both empty means the entry is skipped.
//!
//! Example-compatible server: [qmd](https://github.com/tobi/qmd).
//! QMD exposes `POST /query` as a sessionless REST alias for its MCP `query`
//! tool, avoiding the MCP initialize-handshake overhead.

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
    /// REST query endpoint: `{mcp_url}/query`
    query_endpoint: String,
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
            query_endpoint: format!("{}/query", cfg.mcp_url.trim_end_matches('/')),
        }
    }

    /// Query the remote memory service.
    ///
    /// Returns:
    /// - `None` — service unreachable, timed out, or returned an HTTP error
    ///   (caller should try the next rank in the chain)
    /// - `Some(vec)` — service responded successfully; vec may be empty
    ///   (caller should stop the chain and trust the service's answer)
    pub async fn query(&self, query: &str, limit: usize) -> Option<Vec<MemoryFragment>> {
        // QMD REST format: { searches: [{type: "vec", query: "..."}], limit: N }
        let body = serde_json::json!({
            "searches": [{ "type": "vec", "query": query }],
            "limit": limit
        });

        let resp = match self
            .client
            .post(&self.query_endpoint)
            .json(&body)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::debug!(
                    service = %self.name,
                    rank = self.rank,
                    error = %e,
                    "Memory service unreachable, trying next rank"
                );
                return None;
            }
        };

        // Any non-2xx status means the service is unhealthy — try next rank.
        if !resp.status().is_success() {
            tracing::debug!(
                service = %self.name,
                rank = self.rank,
                status = %resp.status(),
                "Memory service returned error status, trying next rank"
            );
            return None;
        }

        let json: serde_json::Value = match resp.json().await {
            Ok(j) => j,
            Err(e) => {
                tracing::warn!(
                    service = %self.name,
                    "Memory service returned unparseable response: {e}"
                );
                return None;
            }
        };

        // QMD REST response: { results: [{file, title, score, context, snippet}] }
        let results = json
            .get("results")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        let fragments = results
            .into_iter()
            .filter_map(|r| {
                let title = r.get("title").and_then(|v| v.as_str()).unwrap_or("");
                let snippet = r.get("snippet").and_then(|v| v.as_str()).unwrap_or("");

                // Skip entries with no usable content.
                if title.is_empty() && snippet.is_empty() {
                    return None;
                }

                // Combine title + snippet into a single searchable content string.
                let content = match (title.is_empty(), snippet.is_empty()) {
                    (true, _) => snippet.to_string(),
                    (_, true) => title.to_string(),
                    _ => format!("{title}\n\n{snippet}"),
                };

                let score = r
                    .get("score")
                    .and_then(|s| s.as_f64())
                    .unwrap_or(0.5) as f32;
                let file = r
                    .get("file")
                    .and_then(|p| p.as_str())
                    .unwrap_or("")
                    .to_string();
                let docid = r
                    .get("docid")
                    .and_then(|d| d.as_str())
                    .unwrap_or("")
                    .to_string();

                let mut metadata = HashMap::new();
                metadata.insert(
                    "source_path".to_string(),
                    serde_json::Value::String(file),
                );
                metadata.insert(
                    "docid".to_string(),
                    serde_json::Value::String(docid),
                );
                metadata.insert(
                    "mcp_service".to_string(),
                    serde_json::Value::String(self.name.clone()),
                );

                Some(MemoryFragment {
                    id: MemoryId::new(),
                    // External services expose shared knowledge bases, not per-agent memories.
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

        Some(fragments) // Some(empty) = service up but returned no matches
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openfang_types::config::McpMemoryServiceConfig;

    fn make_backend(base_url: &str) -> McpMemoryBackend {
        McpMemoryBackend::new(&McpMemoryServiceConfig {
            name: "test".to_string(),
            mcp_url: base_url.to_string(),
            timeout_ms: 1000,
            rank: 1,
        })
    }

    #[test]
    fn endpoint_uses_query_path() {
        let b = make_backend("http://qmd:8181");
        assert_eq!(b.query_endpoint, "http://qmd:8181/query");
    }

    #[test]
    fn endpoint_strips_trailing_slash() {
        let b = make_backend("http://qmd:8181/");
        assert_eq!(b.query_endpoint, "http://qmd:8181/query");
    }

    /// Verify the response-mapping logic: title + snippet are combined, score
    /// and file are extracted, empty-content entries are dropped.
    #[test]
    fn maps_qmd_results_to_fragments() {
        // Simulate what `query()` does after receiving a valid JSON response.
        let json: serde_json::Value = serde_json::json!({
            "results": [
                {
                    "docid": "#abc123",
                    "file": "notes/arch.md",
                    "title": "Architecture Overview",
                    "score": 0.92,
                    "snippet": "The system uses a layered approach..."
                },
                {
                    "docid": "#def456",
                    "file": "notes/deploy.md",
                    "title": "Deployment Guide",
                    "score": 0.75,
                    "snippet": ""   // empty snippet — title only
                },
                {
                    "docid": "#skip",
                    "file": "empty.md",
                    "title": "",
                    "score": 0.5,
                    "snippet": ""   // both empty — should be skipped
                }
            ]
        });

        let results = json
            .get("results")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        let fragments: Vec<_> = results
            .into_iter()
            .filter_map(|r| {
                let title = r.get("title").and_then(|v| v.as_str()).unwrap_or("");
                let snippet = r.get("snippet").and_then(|v| v.as_str()).unwrap_or("");
                if title.is_empty() && snippet.is_empty() {
                    return None;
                }
                let content = match (title.is_empty(), snippet.is_empty()) {
                    (true, _) => snippet.to_string(),
                    (_, true) => title.to_string(),
                    _ => format!("{title}\n\n{snippet}"),
                };
                let score = r.get("score").and_then(|s| s.as_f64()).unwrap_or(0.5) as f32;
                let file = r.get("file").and_then(|p| p.as_str()).unwrap_or("").to_string();
                Some((content, score, file))
            })
            .collect();

        assert_eq!(fragments.len(), 2, "empty-content entry should be dropped");

        let (content0, score0, file0) = &fragments[0];
        assert!(content0.contains("Architecture Overview"));
        assert!(content0.contains("layered approach"));
        assert!((score0 - 0.92f32).abs() < 0.001);
        assert_eq!(file0, "notes/arch.md");

        let (content1, _score1, _file1) = &fragments[1];
        assert_eq!(content1, "Deployment Guide");
    }
}
