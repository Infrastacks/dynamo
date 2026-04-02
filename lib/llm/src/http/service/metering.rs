// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-request inference metering for InfraStacks billing.
//!
//! Three components:
//! - [`MeteringRecord`] — serializable payload sent to the control plane webhook.
//! - [`MeteringCollector`] — background batch sender (tokio task + mpsc channel).
//! - [`MeteringGuard`] — RAII per-request guard that assembles and submits a record on drop.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

// ---------------------------------------------------------------------------
// MeteringRecord
// ---------------------------------------------------------------------------

/// A single metering event sent to the InfraStacks control plane.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MeteringRecord {
    pub request_id: String,
    pub model: String,
    pub org_id: Option<String>,
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub ttft_ms: Option<f64>,
    pub itl_ms: Option<f64>,
    pub duration_ms: f64,
    pub status: String,
    /// Unix epoch milliseconds.
    pub timestamp: u64,
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct MeteringBatchRequest {
    environment_id: String,
    records: Vec<MeteringRecord>,
}

// ---------------------------------------------------------------------------
// MeteringCollector
// ---------------------------------------------------------------------------

/// Background batch sender that flushes metering records via HTTP POST.
pub struct MeteringCollector {
    tx: mpsc::Sender<MeteringRecord>,
}

impl MeteringCollector {
    /// Create a new collector.
    ///
    /// Spawns a tokio task that batches records and POSTs them as JSON to
    /// `webhook_url` when `batch_size` is reached **or** `flush_interval`
    /// elapses — whichever comes first.
    ///
    /// On HTTP failure the batch is logged and dropped (no retry) to avoid
    /// back-pressure on the inference hot path.
    pub fn new(
        webhook_url: String,
        environment_id: String,
        auth_token: String,
        batch_size: usize,
        flush_interval: Duration,
        cancel: CancellationToken,
    ) -> Arc<Self> {
        let (tx, rx) = mpsc::channel::<MeteringRecord>(4096);

        tokio::spawn(Self::run_loop(
            rx,
            webhook_url,
            environment_id,
            auth_token,
            batch_size,
            flush_interval,
            cancel,
        ));

        Arc::new(Self { tx })
    }

    /// Non-blocking send — drops the record if the channel is full.
    pub fn record(&self, record: MeteringRecord) {
        if let Err(e) = self.tx.try_send(record) {
            tracing::warn!("metering channel full, dropping record: {}", e);
        }
    }

    // -- background task ----------------------------------------------------

    async fn run_loop(
        mut rx: mpsc::Receiver<MeteringRecord>,
        webhook_url: String,
        environment_id: String,
        auth_token: String,
        batch_size: usize,
        flush_interval: Duration,
        cancel: CancellationToken,
    ) {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .expect("failed to build reqwest client for metering");

        let mut batch: Vec<MeteringRecord> = Vec::with_capacity(batch_size);
        let mut interval = tokio::time::interval(flush_interval);
        // The first tick completes immediately — consume it so we don't
        // flush an empty batch right away.
        interval.tick().await;

        loop {
            tokio::select! {
                biased;

                _ = cancel.cancelled() => {
                    // Drain remaining records before exiting.
                    rx.close();
                    while let Some(rec) = rx.recv().await {
                        batch.push(rec);
                    }
                    if !batch.is_empty() {
                        Self::flush(
                            &client,
                            &webhook_url,
                            &environment_id,
                            &auth_token,
                            &mut batch,
                        ).await;
                    }
                    tracing::info!("metering collector shut down");
                    return;
                }

                maybe = rx.recv() => {
                    match maybe {
                        Some(rec) => {
                            batch.push(rec);
                            if batch.len() >= batch_size {
                                Self::flush(
                            &client,
                            &webhook_url,
                            &environment_id,
                            &auth_token,
                            &mut batch,
                        ).await;
                            }
                        }
                        None => {
                            // Channel closed — flush remainder and exit.
                            if !batch.is_empty() {
                                Self::flush(
                            &client,
                            &webhook_url,
                            &environment_id,
                            &auth_token,
                            &mut batch,
                        ).await;
                            }
                            return;
                        }
                    }
                }

                _ = interval.tick() => {
                    if !batch.is_empty() {
                        Self::flush(
                            &client,
                            &webhook_url,
                            &environment_id,
                            &auth_token,
                            &mut batch,
                        ).await;
                    }
                }
            }
        }
    }

    async fn flush(
        client: &reqwest::Client,
        url: &str,
        environment_id: &str,
        auth_token: &str,
        batch: &mut Vec<MeteringRecord>,
    ) {
        let count = batch.len();
        let payload = MeteringBatchRequest {
            environment_id: environment_id.to_string(),
            records: batch.clone(),
        };

        match client
            .post(url)
            .bearer_auth(auth_token)
            .json(&payload)
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {
                tracing::debug!("metering: flushed {} records", count);
            }
            Ok(resp) => {
                tracing::warn!(
                    "metering: webhook returned {} — dropping {} records",
                    resp.status(),
                    count
                );
            }
            Err(e) => {
                tracing::warn!("metering: POST failed ({}) — dropping {} records", e, count);
            }
        }
        batch.clear();
    }
}

// ---------------------------------------------------------------------------
// MeteringGuard
// ---------------------------------------------------------------------------

/// RAII guard that assembles a [`MeteringRecord`] on drop and forwards it to
/// the [`MeteringCollector`].
pub struct MeteringGuard {
    collector: Arc<MeteringCollector>,
    request_id: String,
    model: String,
    org_id: Option<String>,
    start_time: Instant,
    input_tokens: AtomicUsize,
    output_tokens: AtomicUsize,
    ttft_ms: Mutex<Option<f64>>,
    status: Mutex<String>,
}

impl MeteringGuard {
    pub fn new(
        collector: Arc<MeteringCollector>,
        request_id: String,
        model: String,
        org_id: Option<String>,
    ) -> Self {
        Self {
            collector,
            request_id,
            model,
            org_id,
            start_time: Instant::now(),
            input_tokens: AtomicUsize::new(0),
            output_tokens: AtomicUsize::new(0),
            ttft_ms: Mutex::new(None),
            status: Mutex::new("error".to_string()),
        }
    }

    /// Accumulate token counts (called per chunk).
    pub fn record_tokens(&self, input: usize, output: usize) {
        self.input_tokens.fetch_add(input, Ordering::Relaxed);
        self.output_tokens.fetch_add(output, Ordering::Relaxed);
    }

    /// Record time-to-first-token (milliseconds).
    pub fn record_ttft(&self, ms: f64) {
        if let Ok(mut guard) = self.ttft_ms.lock() {
            *guard = Some(ms);
        }
    }

    /// Mark the request as successful.
    pub fn mark_ok(&self) {
        if let Ok(mut guard) = self.status.lock() {
            *guard = "ok".to_string();
        }
    }

    /// Mark the request as failed with a custom status string.
    pub fn mark_error(&self, status: &str) {
        if let Ok(mut guard) = self.status.lock() {
            *guard = status.to_string();
        }
    }
}

impl Drop for MeteringGuard {
    fn drop(&mut self) {
        let duration_ms = self.start_time.elapsed().as_secs_f64() * 1000.0;
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let ttft_ms = self.ttft_ms.lock().ok().and_then(|g| *g);
        let status = self
            .status
            .lock()
            .ok()
            .map(|g| g.clone())
            .unwrap_or_else(|| "unknown".to_string());

        let record = MeteringRecord {
            request_id: self.request_id.clone(),
            model: self.model.clone(),
            org_id: self.org_id.clone(),
            input_tokens: self.input_tokens.load(Ordering::Relaxed),
            output_tokens: self.output_tokens.load(Ordering::Relaxed),
            ttft_ms,
            itl_ms: None, // populated by metrics integration if available
            duration_ms,
            status,
            timestamp,
        };

        self.collector.record(record);
    }
}

#[cfg(test)]
mod tests {
    use super::{MeteringBatchRequest, MeteringRecord};

    #[test]
    fn serializes_metering_batch_in_control_plane_shape() {
        let payload = MeteringBatchRequest {
            environment_id: "env_test".to_string(),
            records: vec![MeteringRecord {
                request_id: "req_123".to_string(),
                model: "meta-llama/Llama-3.1-8B-Instruct".to_string(),
                org_id: Some("org_123".to_string()),
                input_tokens: 128,
                output_tokens: 64,
                ttft_ms: Some(42.0),
                itl_ms: Some(7.5),
                duration_ms: 512.0,
                status: "ok".to_string(),
                timestamp: 1_750_000_000_000,
            }],
        };

        let value = serde_json::to_value(payload).expect("payload should serialize");

        assert_eq!(value["environmentId"], "env_test");
        assert_eq!(value["records"][0]["requestId"], "req_123");
        assert_eq!(
            value["records"][0]["orgId"],
            serde_json::Value::String("org_123".to_string())
        );
        assert_eq!(value["records"][0]["inputTokens"], 128);
        assert_eq!(value["records"][0]["outputTokens"], 64);
        assert_eq!(value["records"][0]["durationMs"], 512.0);
        assert_eq!(value["records"][0]["status"], "ok");
    }
}
