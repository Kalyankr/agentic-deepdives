# 🃏 Feature & Embedding Store — One-Page Cheat-Sheet

> Last-minute recall card for the [full HLD](README.md). Drill the bold bits.

## The one idea
The **data layer for ML**: define a feature once, serve it **consistently** for **offline training** (point-in-time-correct) and **online inference** (ms key lookup). Its real product is **two guarantees: point-in-time correctness + train/serve consistency.** Looks up **by key** (vs the [vector DB](../vector-database/README.md)'s search **by similarity**).

## The two guarantees (lead with these)
- **Point-in-time correctness** — join each label to the latest feature value with **`ts ≤ event_time`** (within TTL). Future values = **label leakage** → inflated offline metrics, online failure.
- **Train/serve consistency** — **one definition, one transform** feeds both stores → no **train/serve skew** (the silent #1 production ML bug).

## Dual store (the architecture)
| | **Online store** | **Offline store** |
|---|---|---|
| Holds | **latest** value/(entity,feat) | **all history + timestamps** |
| Engine | sharded KV (Redis/Dynamo/Cassandra) | warehouse/lake (BigQuery/Parquet) |
| Powers | **inference** (ms key lookup) | **training** (point-in-time join) |
| Size | small (~entities×features) | huge (×history → 100s TB–PB) |

Fed by **one materialization** pipeline + a **registry** → consistency.

## Why not just a DB/cache?
KV gives latest fast but **no time-travel** ("what was this at label time?") and **no one-definition guarantee**. Those two are the whole point.

## Point-in-time / as-of join
For each `(entity, event_ts)` label row → **latest feature value with `feature_ts ≤ event_ts`** within TTL, else null. Never a future value. The **expensive** part of training-set gen → partition by time, **bucket/co-partition by entity** (kill the shuffle), push into the engine, incremental.

## Kill train/serve skew
- **One transform** feeds offline + online (no re-impl in serving).
- **Backfill** new features by replaying the same transform over history.
- **On-demand features** = danger zone → **same registered function** offline + online.
- **Audit:** log online-served values, **compare to offline** for same (entity,time) → skew detector.

## Online store
Sharded KV, key=`(entity_id, feature_view)`, value=latest row (embeddings as `float[]`). **Latest-only** (bounded). **Multi-get** for fan-out, **co-locate a view's columns in one value**, shard by entity, **replicate hot keys**. p99 < 10–50 ms.

## Offline store
Columnar warehouse/lake, **every value with `event_ts`**, partitioned by date. Historical retrieval = **as-of join** per view. Scale: time-prune, bucket by key, co-partition, window functions / Spark.

## Materialization (batch + streaming)
**Batch** (hourly/daily): aggregates/slow features → load latest online + append history offline. **Streaming** (Kafka): windowed features → **upsert online in seconds**. **Idempotent `(entity, ts)` upserts.** Size for the **update rate** (dual writes).

## Transformations
**Batch** (low skew) · **streaming** (medium — reconcile) · **on-demand/request-time** (high skew — one function both paths). Keep **deterministic + versioned**.

## Embeddings as features vs vector DB
**Feature store** = fetch **by key** ("user 42's embedding") to **rank** a known set. **Vector DB** = search **by similarity** to **retrieve**. They **compose**: `VDB ANN retrieve → FS multi-get features → rank`. FS produces embeddings, can **publish** them into the VDB.

## Lambda vs kappa
**Lambda** = batch (accurate, late) + streaming (fresh, approx) → can disagree → share logic + **batch corrects streaming**. **Kappa** = stream-only (reprocess from log) → one code path, no skew.

## Numbers (state assumptions)
- Online: 100M users × 1000 feat × 8 B ≈ **800 GB** (+items) → ~1 TB hot, sharded KV.
- Embeddings: 100M × 256-dim fp32 (1 KB) ≈ **100 GB**.
- Fan-out: 10K req/s × (1 user + 500 items) ≈ **5M reads/s** → multi-get is mandatory.
- Offline: history → **100s TB–PB** (≫ online).

## Registry, freshness, monitoring
**Registry:** definitions + lineage + **versioning** (new version on change → no silent breakage) + discovery/reuse. **Freshness:** SLA per feature (streaming ~secs, batch ~hours); TTL → null past it. **Monitor:** freshness lag · drift (PSI/KL) · coverage/null · **skew (online vs offline)**.

## Cost order
TTL/evict cold online entities → **batch over streaming** where possible → **reuse** (share one feature across models) → tier hot RAM / cold object store → co-locate columns → incremental materialization. Track **$/online read** + **$/feature-month**.

## Top failure modes
**label leakage** (as-of join) · **train/serve skew** (one def + audit) · online tail (multi-get, hot-key replicate) · **lambda disagreement** · slow point-in-time join (partition/bucket) · stale features (freshness SLA) · drift · null-in-prod (coverage monitor) · on-demand skew · schema change breaks model (**version**).

---
[← HLD](README.md) · [Q&A](questions.md) · [Answers](answers.md) · [Index](../../README.md)
