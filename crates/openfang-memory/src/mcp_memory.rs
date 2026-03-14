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
    /// REST ingest endpoint: `{mcp_url}/ingest`
    ingest_endpoint: String,
    /// Health probe endpoint: `{mcp_url}/health`
    health_endpoint: String,
}

impl McpMemoryBackend {
    /// Create a backend from its config, building a dedicated `reqwest::Client`
    /// with the configured timeout baked in.
    pub fn new(cfg: &McpMemoryServiceConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(cfg.timeout_ms))
            .build()
            .expect("reqwest client build failed");
        let base = cfg.mcp_url.trim_end_matches('/');
        Self {
            client,
            name: cfg.name.clone(),
            rank: cfg.rank,
            query_endpoint: format!("{base}/query"),
            ingest_endpoint: format!("{base}/ingest"),
            health_endpoint: format!("{base}/health"),
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
        // QMD REST format: send both vec (semantic) and lex (keyword) searches.
        // Freshly ingested memories are BM25-indexed immediately but not yet embedded,
        // so lex search ensures they are found before `qmd embed` runs.
        let body = serde_json::json!({
            "searches": [
                { "type": "vec", "query": query },
                { "type": "lex", "query": query },
            ],
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

    /// Push one memory fragment to the remote service via `POST /ingest`.
    ///
    /// Returns `true` if the service accepted the fragment (HTTP 2xx), `false` on
    /// any error (offline, timeout, non-2xx). Failure is non-fatal — the caller
    /// falls back to keeping the fragment in SQLite for the next backfill cycle.
    pub async fn push(&self, fragment: &MemoryFragment) -> bool {
        let content = format_fragment_as_markdown(fragment);
        let path = format!(
            "{}/{}/{}.md",
            fragment.agent_id.0,
            fragment.created_at.format("%Y-%m"),
            fragment.id.0,
        );

        let body = serde_json::json!({
            "collection": "openfang-memories",
            "path":       path,
            "content":    content,
            "created_at": fragment.created_at.to_rfc3339(),
        });

        match self.client.post(&self.ingest_endpoint).json(&body).send().await {
            Ok(r) => {
                if r.status().is_success() {
                    tracing::debug!(
                        service = %self.name,
                        id      = %fragment.id.0,
                        "Memory fragment pushed to MCP service"
                    );
                    true
                } else {
                    tracing::debug!(
                        service = %self.name,
                        status  = %r.status(),
                        "MCP push returned non-2xx, will retry on next sync"
                    );
                    false
                }
            }
            Err(e) => {
                tracing::debug!(
                    service = %self.name,
                    error   = %e,
                    "MCP push failed (service offline?), will retry on next sync"
                );
                false
            }
        }
    }

    /// Returns the health-probe URL for this service (`{mcp_url}/health`).
    pub fn health_endpoint(&self) -> &str {
        &self.health_endpoint
    }
}

/// Convert an OpenFang `MemoryFragment` into a QMD-indexable markdown document.
///
/// The resulting markdown is immediately BM25-searchable after `insertDocument`
/// and becomes vector-searchable after the next `qmd embed` run.
fn format_fragment_as_markdown(f: &MemoryFragment) -> String {
    let title = f
        .content
        .lines()
        .next()
        .unwrap_or("Memory Fragment")
        .chars()
        .take(60)
        .collect::<String>();

    let source = serde_json::to_string(&f.source)
        .unwrap_or_else(|_| "\"unknown\"".to_string())
        .trim_matches('"')
        .to_string();

    format!(
        "# {title}\n\n\
         **Agent**: {agent}\n\
         **Source**: {source}\n\
         **Scope**: {scope}\n\
         **Confidence**: {confidence:.2}\n\
         **Created**: {created_at}\n\n\
         {content}\n",
        title = title,
        agent = f.agent_id.0,
        source = source,
        scope = f.scope,
        confidence = f.confidence,
        created_at = f.created_at.to_rfc3339(),
        content = f.content,
    )
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

    #[test]
    fn ingest_endpoint_uses_ingest_path() {
        let b = make_backend("http://qmd:8181");
        assert_eq!(b.ingest_endpoint, "http://qmd:8181/ingest");
    }

    #[test]
    fn health_endpoint_uses_health_path() {
        let b = make_backend("http://qmd:8181");
        assert_eq!(b.health_endpoint(), "http://qmd:8181/health");
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

    #[test]
    fn format_fragment_has_title_and_metadata() {
        use openfang_types::{agent::AgentId, memory::{MemoryId, MemorySource}};
        use std::collections::HashMap;

        let frag = MemoryFragment {
            id: MemoryId(uuid::Uuid::nil()),
            agent_id: AgentId(uuid::Uuid::nil()),
            content: "The user prefers concise answers.".to_string(),
            embedding: None,
            metadata: HashMap::new(),
            source: MemorySource::Conversation,
            confidence: 0.9,
            created_at: chrono::DateTime::parse_from_rfc3339("2026-03-14T00:00:00Z")
                .unwrap()
                .with_timezone(&chrono::Utc),
            accessed_at: chrono::Utc::now(),
            access_count: 0,
            scope: "episodic".to_string(),
        };

        let md = format_fragment_as_markdown(&frag);
        assert!(md.starts_with("# The user prefers concise answers."));
        assert!(md.contains("**Source**: Conversation"));
        assert!(md.contains("**Scope**: episodic"));
        assert!(md.contains("**Confidence**: 0.90"));
        assert!(md.contains("The user prefers concise answers."));
    }

    #[test]
    fn format_fragment_title_truncated_at_60() {
        use openfang_types::{agent::AgentId, memory::{MemoryId, MemorySource}};
        use std::collections::HashMap;

        let long_line = "A".repeat(80);
        let frag = MemoryFragment {
            id: MemoryId(uuid::Uuid::nil()),
            agent_id: AgentId(uuid::Uuid::nil()),
            content: long_line.clone(),
            embedding: None,
            metadata: HashMap::new(),
            source: MemorySource::System,
            confidence: 1.0,
            created_at: chrono::Utc::now(),
            accessed_at: chrono::Utc::now(),
            access_count: 0,
            scope: "global".to_string(),
        };

        let md = format_fragment_as_markdown(&frag);
        let first_line = md.lines().next().unwrap();
        // "# " prefix + 60 chars = 62 chars
        assert_eq!(first_line.len(), 62, "title should be truncated to 60 chars");
    }
}
