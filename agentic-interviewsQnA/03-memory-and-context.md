# 03 — Memory & Context

> Goal: design memory for agents — short vs. long term, the episodic/semantic/procedural taxonomy, RAG, and "context engineering" (the real skill).

---

## 3.1 Why agents need memory

An LLM is **stateless** — each call only knows what's in its context window. Memory is everything you add to make the agent **stateful** across steps and sessions: prior messages, retrieved facts, learned procedures, user preferences.

> 💡 Memory = the bridge between a stateless model and a coherent, personalized, multi-step agent.

---

## 3.2 Short-term vs. long-term memory

| | Short-term (working) | Long-term (persistent) |
|---|----------------------|------------------------|
| **Lives in** | The context window | External store (vector DB, SQL, KV, files) |
| **Scope** | Current task/session | Across sessions/users |
| **Examples** | Recent turns, scratchpad, current plan, tool outputs | User profile, past conversations, knowledge base, learned skills |
| **Limit** | Token budget | Storage + retrieval quality |
| **Failure** | Overflow → truncation/forgetting | Bad retrieval → irrelevant/missing context |

Short-term memory management techniques:
- **Sliding window** — keep last N turns.
- **Summarization / compaction** — periodically compress old turns into a summary (LangGraph/Claude "context compaction").
- **Scratchpad** — a structured working-memory blob the agent reads/writes.
- **Token-budget eviction** — drop or summarize lowest-value items when near the limit.

---

## 3.3 The memory taxonomy (cognitive analogy)

A clean framework interviewers like (borrowed from human memory):

| Type | What it stores | Agent example | Typical store |
|------|----------------|---------------|---------------|
| **Episodic** | Specific past experiences/events | "Last time, the user preferred bullet summaries"; past trajectories | Vector DB of episodes |
| **Semantic** | Facts & knowledge | Domain knowledge, user profile, entities | Vector DB / knowledge graph |
| **Procedural** | How to do things | Tool-use skills, system prompt, learned routines | Code, prompt, skill library |

(Some add **working memory** = the in-context scratchpad.)

💡 Reflexion stores **episodic** self-reflections. A user-profile store is **semantic**. A library of reusable skills/prompts is **procedural**.

---

## 3.4 Retrieval-Augmented Generation (RAG) — the workhorse of long-term memory

RAG = retrieve relevant external knowledge at query time and inject it into the prompt, so the model is grounded in current/private data it wasn't trained on.

**Pipeline:**
```
Ingest:  documents → chunk → embed → index (vector DB)
Query:   user query → embed → similarity search → top-k chunks
Augment: stuff chunks into prompt → LLM generates grounded answer
```

**Design knobs to discuss:**
- **Chunking** — size/overlap; semantic vs. fixed; too big = noise, too small = lost context.
- **Embeddings** — model choice, dimensionality, domain fit.
- **Retrieval** — dense (vectors) vs. sparse (BM25) vs. **hybrid**; top-k; metadata filters.
- **Re-ranking** — cross-encoder reranker over top-k to boost precision.
- **Query transformation** — HyDE, multi-query, query rewriting/decomposition.
- **Grounding/citations** — return sources; reduce hallucination.

**Agentic RAG** (vs. naive RAG): the agent *decides* whether to retrieve, *what* to query, *which* source/tool to use, and can **iterate** (retrieve → reason → retrieve again). Retrieval becomes a **tool the agent calls**, not a fixed pre-step.

⚠️ Common interview trap: treating RAG as a fixed pipeline only. Senior answer mentions agentic/iterative retrieval, hybrid search, reranking, and evaluation (faithfulness, context precision/recall, e.g. RAGAS).

---

## 3.5 Memory stores & when to use them

| Store | Best for | Note |
|-------|----------|------|
| **Vector DB** (Pinecone, Weaviate, pgvector, Chroma, Milvus) | Semantic similarity search | Core of RAG & episodic memory |
| **Relational/KV** (Postgres, Redis) | Structured state, user profiles, sessions | Fast exact lookups |
| **Knowledge graph** (Neo4j) | Entities + relationships, multi-hop | GraphRAG for connected reasoning |
| **Document/file store** | Artifacts, large blobs | Agent "filesystem" memory |
| **Full-text/search** (Elastic, BM25) | Keyword precision | Pair with vectors → hybrid |

---

## 3.6 Context engineering (the real skill)

💡 Beyond prompt engineering: **context engineering is curating exactly what goes into the window at each step** — instructions, relevant memory, tools, retrieved data, and current state — within a finite token budget.

The four failure modes of long context to name:
- **Context poisoning** — a hallucination/error gets stored and reused.
- **Context distraction** — too much irrelevant info dilutes attention.
- **Context confusion** — conflicting/duplicated info.
- **Context clash** — newly retrieved info contradicts what's already there.

Tactics:
- **Retrieve, don't stuff** — pull only what's needed, when needed.
- **Compress/summarize** old turns; keep a rolling summary.
- **Isolate** — give sub-agents their own clean context (a big reason for multi-agent designs).
- **Structure** — use clear sections, delimiters, and schemas; put the most important info at the start/end (recency/primacy).
- **Offload to memory/files** — store large artifacts externally, reference by handle.

⚠️ "Lost in the middle": models attend less to info buried in the middle of a long context. Put critical instructions/data at the edges.

---

## 3.7 Putting it together — a memory architecture

```
                ┌────────────────────────────┐
   user turn ──▶│   Working memory (context) │◀── rolling summary
                │  system + recent turns +   │
                │  scratchpad + retrieved ctx │
                └─────┬───────────────┬──────┘
        write/recall  │               │  retrieve top-k
                ┌─────▼─────┐   ┌──────▼───────┐
                │ Episodic  │   │  Semantic    │
                │ (past     │   │ (facts, user │
                │ episodes) │   │  profile, KB)│
                └───────────┘   └──────────────┘
                ┌────────────────────────────┐
                │ Procedural (skills/prompts) │
                └────────────────────────────┘
```

Decisions to articulate: **what to write** (write policy), **when to retrieve** (read policy), **how to forget** (eviction/decay), and **how to keep memory consistent** (avoid poisoning).

---

## Interview questions for this chapter

1. Differentiate short-term and long-term memory in agents and how you manage each. *(3.2)*
2. Explain episodic vs. semantic vs. procedural memory with agent examples. *(3.3)*
3. Walk through a RAG pipeline and the main quality knobs. *(3.4)*
4. What's *agentic* RAG and how does it differ from naive RAG? *(3.4)*
5. Define context engineering and three of its failure modes. *(3.6)*
6. The context window keeps overflowing on long sessions. What do you do? *(3.2, 3.6)*
7. How do you evaluate a RAG system's quality? *(3.4, see Ch 08)*

**Model answer to #6:** Layer strategies: rolling summarization/compaction of old turns, sliding window for raw turns, offload large artifacts to external memory and reference by handle, retrieve only relevant context per step instead of carrying everything, and split work across sub-agents with isolated contexts. Add token-budget accounting so eviction is deliberate, and keep critical instructions pinned at the start/end of the window.
