# Stage 4 — Answer Key (Evaluation)

> Full worked answers to [interview-questions.md](interview-questions.md). The bar here is treating evaluation as **experiment design with uncertainty**, distrusting benchmarks intelligently, separating sub-system failures, and knowing the statistics.

---

## 🟢 Fundamentals

**1. Intrinsic vs extrinsic evaluation.**
**Intrinsic** measures a model property in isolation (perplexity, BLEU, accuracy on a benchmark). **Extrinsic** measures performance on the real downstream task/goal it serves (user task success, retention, revenue, ticket-resolution rate). Intrinsic is cheap and fast; extrinsic is what actually matters — they often disagree.

**2. What perplexity measures + limits.**
Perplexity is the exponentiated per-token cross-entropy — the model's average uncertainty/branching factor on held-out text. Limits: it only measures *language-modeling fit*, not helpfulness/correctness/safety; it's **not comparable across tokenizers or datasets**; and a low-PPL model can still hallucinate, refuse, or fail instructions.

**3. BLEU/ROUGE + when they mislead.**
They measure **n-gram overlap** with reference text (BLEU = precision-oriented, for translation; ROUGE = recall-oriented, for summarization). They mislead because they reward surface overlap, not meaning: a correct paraphrase with different words scores low, and a fluent-but-wrong answer sharing n-grams scores high. Poor for open-ended generation where many valid answers exist.

**4. pass@k and why trustworthy.**
For code, generate $k$ samples and count the problem solved if **any** passes the unit tests. It's trustworthy because it's grounded in an **objective, executable** signal (tests either pass or fail) rather than surface similarity — no judge or reference-matching ambiguity.

**5. Three benchmarks + what each tests.**
- **MMLU:** broad multiple-choice knowledge/reasoning across 57 subjects.
- **HumanEval / MBPP:** functional code generation (pass@k via unit tests).
- **GSM8K / MATH:** multi-step mathematical reasoning.
(Others: MT-Bench/Chatbot Arena for chat quality, HELM for holistic, TruthfulQA for truthfulness.)

**6. Benchmark contamination.**
When test items (or near-duplicates) appear in the model's training data, so it has effectively memorized answers — inflating scores without real capability. The central threat to benchmark validity, worsened by web-scale pretraining.

**7. LLM-as-a-judge.**
Using a strong LLM to score/compare other models' outputs against a rubric or pairwise. Scalable and cheap vs humans, decent agreement on many tasks — but carries biases (position, length, self-preference) that must be controlled.

**8. Why human eval is still needed.**
For subjective/open-ended quality, safety nuance, and as the **ground truth** to validate automatic metrics and judges. Humans catch failure modes (subtle hallucination, tone, harm) that metrics miss, and are the anchor that tells you whether your cheap proxies are trustworthy.

**9. Hallucination as an eval problem.**
A hallucination is fluent, confident output that is **factually wrong or unsupported**. It's hard to evaluate because surface metrics (BLEU/perplexity) don't detect it and it requires checking factual grounding — you need faithfulness/attribution metrics, reference-based fact checks, or human/judge verification against sources.

**10. RAG — two things to evaluate separately.**
**Retrieval** (did we fetch the right documents? recall@k, MRR, nDCG) and **generation** (given the retrieved context, is the answer faithful and relevant? faithfulness/groundedness + answer-relevance). Separating them tells you *which* sub-system to fix.

---

## 🟡 Core (L4–L5)

**11. Top MMLU yet fails in production — why.**
- **Contamination:** memorized the benchmark, not the skill.
- **Distribution shift:** production prompts/format/domain differ from the clean multiple-choice benchmark.
- **Metric–goal mismatch:** MMLU tests knowledge recall, not multi-turn helpfulness, instruction-following, tool use, latency, or safety.
- **Format sensitivity:** real users phrase things messily; benchmarks are tidy.
- **No grounding/hallucination** handling, refusal behavior, or tone — none captured by MMLU.

**12. LLM-judge biases + mitigations.**
- **Position bias** (favors first/second) → **swap order and average** both directions.
- **Verbosity/length bias** (favors longer) → length-control, instruct to ignore length, or normalize.
- **Self-preference** (favors its own family's style) → use a different judge family / ensemble.
- **Style/sycophancy bias** (formatting, confidence) → rubric-based scoring, blind to style.
- **Position of correct answer / label bias** in MCQ → randomize options.
Always **validate against human labels** and report agreement.

**13. Detect train↔benchmark contamination.**
**N-gram / substring overlap** (e.g. 13-gram match) or **MinHash/LSH** between benchmark items and the training corpus; flag and quantify hits. Complement with **behavioral tests**: compare performance on the original benchmark vs perturbed/freshly-written equivalents (a big drop ⇒ memorization), and embed **canary strings** to detect leakage.

**14. HELM's philosophy.**
**Holistic** evaluation: assess many models on many scenarios across **multiple dimensions** (accuracy, robustness, fairness, bias, toxicity, efficiency, calibration) with standardized prompting — instead of a single accuracy number. It matters because one metric hides tradeoffs; a model can be accurate but biased, slow, or miscalibrated.

**15. Why pairwise over absolute scoring.**
Humans and judges are **noisy and miscalibrated** on absolute scales (one rater's 7/10 ≠ another's), but **consistent at relative judgments** ("A is better than B"). Pairwise comparisons have higher inter-rater agreement, cancel scale drift, and aggregate cleanly (Elo/Bradley–Terry) into rankings.

**16. Retrieval vs generation quality in RAG.**
**Retrieval:** with gold-document labels, compute recall@k, precision@k, MRR, nDCG — does the right context get fetched? **Generation:** given the retrieved context, measure **faithfulness/groundedness** (is every claim supported by context?) and **answer relevance** (does it address the question?), plus end-to-end correctness. Measuring both lets you attribute errors.

**17. Calibration — what and why.**
Calibration = the model's stated/implied confidence matches its empirical accuracy (of things it says with 80% confidence, ~80% are right). It matters because downstream systems and users act on confidence — a miscalibrated, overconfident model causes unwarranted trust and bad abstention/routing decisions. Measured with ECE / reliability diagrams.

**18. Goodhart's law in eval.**
"When a measure becomes a target, it ceases to be a good measure." Optimizing directly for a benchmark (or judge) produces models that **game the metric** without improving real quality — e.g. training on benchmark-like data, padding length for a length-biased judge. Guard with held-out/rotating evals and multiple metrics.

**19. Reporting results honestly.**
Report **confidence intervals** (bootstrap), **multiple seeds**/decoding settings with variance, **significance tests** for comparisons (not just point estimates), test-set size and provenance, **contamination checks**, and prompt/format details for reproducibility. State limitations; avoid cherry-picked single runs.

**20. Offline benchmarks vs online/production evals.**
**Offline:** fixed datasets, fast, reproducible, cheap — good for regression gating, but static and prone to contamination/distribution gap. **Online:** real-traffic A/B tests and product metrics (success, engagement, complaints) — the ground truth for impact, but slow, confounded, and ethically/operationally constrained. Use offline to gate, online to confirm.

---

## 🔴 Senior / Staff deep dives

**21. Eval with no ground-truth labels (open-ended summarization).**
Combine signals: (1) **rubric-based LLM-judge** scoring concrete dimensions (faithfulness, coverage, conciseness) with order-swapping; (2) **pairwise preference** between systems (Elo) — more reliable than absolute; (3) **reference-free metrics** (e.g. summarization faithfulness/QA-based consistency); (4) **human spot-checks** on a sample to **validate the judge** (report judge–human agreement / Cohen's κ); (5) **report CIs**. The key is triangulation + measuring how much you can trust each automatic proxy.

**22. End-to-end eval before launch (customer assistant).**
- **Capability benchmarks** (task-specific accuracy, instruction-following).
- **Faithfulness/hallucination** evals (esp. if RAG).
- **Safety/red-team**: jailbreaks, toxicity, PII, policy violations.
- **Robustness**: prompt-format/perturbation sensitivity.
- **Latency/cost** under realistic load (p50/p95, $/req).
- **A/B + online metrics**: success rate, escalation, satisfaction.
- A **regression suite** + contamination policy so future changes are gated.
Define launch thresholds per dimension up front.

**23. Offline up, users unhappy — what's wrong with the eval.**
Likely: **contamination** (offline gains aren't real), **metric–goal mismatch / Goodhart** (optimized a proxy users don't care about), **distribution shift** (test set ≠ real traffic), **prompt-format sensitivity** (eval prompts differ from product), **stale/tiny test set**, or **judge bias** flattering the new model. Fix by aligning eval data/format to production, adding online metrics, refreshing/holding out data, and validating judges against humans.

**24. Validate & de-bias an LLM-judge for a leaderboard.**
- **Position bias:** evaluate both orders (A,B) and (B,A), average — discard inconsistent pairs.
- **Length bias:** control/normalize length or instruct to ignore it.
- **Self-preference:** use a judge from a different family, or an **ensemble** of judges.
- **Calibrate vs humans:** collect human labels on a sample, report **agreement (Cohen's/Fleiss' κ)**; only trust the judge where agreement is high.
- **Report uncertainty** (CIs) and keep prompts fixed/audited. Publish the validation, not just the ranking.

**25. Contamination-resistant eval program.**
- **Private held-out sets** never shared or posted online.
- **Time-split / freshly-authored** evals created *after* the training cut-off.
- **Rotating** benchmarks so a fixed set can't be optimized against.
- **Canary strings** seeded to detect leakage.
- **N-gram/MinHash overlap scans** of training data vs every eval set, with reported overlap.
- Governance: separate the people who curate evals from those who curate training data.

**26. Evaluate RAG to localize failures.**
Build an **error-attribution matrix**: independently score **retrieval** (recall@k/MRR/nDCG vs gold docs) and **generation** (faithfulness + answer-relevance given the *actual* retrieved context). Then classify each failure: retrieval miss (right doc not fetched) vs generation failure (right context, wrong/unfaithful answer) vs both. This tells you whether to fix the retriever (chunking, embeddings, k) or the generator (prompt, grounding, model).

**27. Statistically decide A beats B.**
Use a **paired** comparison (same items for both models) → McNemar's test or a paired bootstrap on the score difference; report a **confidence interval and effect size**, not just a p-value. Check **power/sample size** (small test sets can't detect small gaps). Apply **multiple-comparison correction** if testing many slices/models. Conclude "A beats B" only if the CI on the difference excludes 0 and the effect is practically meaningful.

**28. Critique a benchmark (e.g. MMLU).**
MMLU is **multiple-choice**, so it rewards recognition over generation, is **contamination-prone** (widely on the web), has some **mislabeled/ambiguous items**, tests **static knowledge** not reasoning/tool-use/multi-turn, and is **format-sensitive** (scores swing with prompting). It says little about helpfulness, safety, hallucination, or real-task success — necessary but far from sufficient.

---

## 🧮 Stats & metrics

**29. precision@k, recall@k, MRR, nDCG.**
- **precision@k** = (relevant in top k)/k — how clean the top results are.
- **recall@k** = (relevant in top k)/(total relevant) — did we find them? Use when you must retrieve all relevant docs (RAG recall).
- **MRR** = mean of $1/\text{rank}$ of the **first** relevant result — good when one correct answer and rank matters.
- **nDCG** = discounted cumulative gain normalized by ideal ordering — best when there are **graded relevance** levels and ranking order matters.

**30. Bootstrap CI for accuracy.**
Resample the $n$ per-item correctness values **with replacement** $n$ times to form a bootstrap sample; compute its accuracy; repeat $B$ (e.g. 1000–10000) times to get a distribution of the metric; the 2.5th and 97.5th percentiles give a **95% CI**. No normality assumption; works for any metric.

**31. 71% vs 73% on 500 items — significant?**
Probably not. Rough SE of a proportion ≈ $\sqrt{p(1-p)/n}\approx\sqrt{0.72\cdot0.28/500}\approx 2.0\%$; the difference (2 points) is ~1 SE — well within noise. Properly, use a **paired** test (McNemar's, since same items) or a paired bootstrap on the per-item differences; here it would **fail to reach significance**. Need a larger/paired test or more items.

**32. Unbiased pass@k from n>k samples (sketch).**
Naively estimating pass@k by sampling exactly $k$ is high-variance. Instead draw $n\ge k$ samples, count $c$ that pass, and use the **unbiased estimator**
$$\text{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}},$$
i.e. 1 minus the probability that a random size-$k$ subset contains **no** passing sample (hypergeometric). Averaging over problems gives a low-variance, unbiased estimate.

---

## 💻 Coding / implementation

**33. Pairwise judge controlling position bias.**
```python
def judge_pair(judge, prompt, a, b):
    def ask(x, y):  # returns 'A' or 'B'
        return judge(f"Question: {prompt}\n\n[A]: {x}\n\n[B]: {y}\n"
                     "Which is better? Answer A or B.")
    r1 = ask(a, b)                  # a as A
    r2 = ask(b, a)                  # a as B (swapped)
    a_wins = (r1 == 'A') + (r2 == 'B')   # count a's wins across both orders
    if a_wins == 2:  return 'a'
    if a_wins == 0:  return 'b'
    return 'tie'                    # inconsistent across order -> tie
```
Swapping and requiring agreement cancels position bias.

**34. pass@k estimation + tests.**
```python
from math import comb
def pass_at_k(n, c, k):
    if n - c < k: return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)

# unit-test driven scoring
def score_problem(samples, tests, k):
    c = sum(all(run(s, t) for t in tests) for s in samples)
    return pass_at_k(len(samples), c, k)

assert pass_at_k(10, 0, 1) == 0.0
assert pass_at_k(10, 10, 1) == 1.0
assert 0 < pass_at_k(10, 1, 5) < 1
```

**35. N-gram overlap contamination scanner.**
```python
def ngrams(text, n=13):
    toks = text.split()
    return {" ".join(toks[i:i+n]) for i in range(len(toks) - n + 1)}

def contamination(train_docs, bench_docs, n=13):
    train_ng = set().union(*(ngrams(d, n) for d in train_docs))
    flagged = []
    for i, b in enumerate(bench_docs):
        hits = ngrams(b, n) & train_ng
        if hits:
            flagged.append((i, len(hits)))
    return flagged          # benchmark items with verbatim overlap
```
(Production: hash n-grams / MinHash for scale.)

**36. recall@k and MRR.**
```python
def recall_at_k(retrieved, gold, k):
    topk = set(retrieved[:k])
    return len(topk & set(gold)) / max(1, len(gold))

def mrr(retrieved, gold):
    for rank, doc in enumerate(retrieved, start=1):
        if doc in gold:
            return 1.0 / rank
    return 0.0
```

**37. Bootstrap CI for any metric.**
```python
import numpy as np
def bootstrap_ci(per_item, metric_fn, B=10000, alpha=0.05):
    per_item = np.asarray(per_item); n = len(per_item)
    stats = [metric_fn(per_item[np.random.randint(0, n, n)]) for _ in range(B)]
    lo, hi = np.percentile(stats, [100*alpha/2, 100*(1-alpha/2)])
    return metric_fn(per_item), (lo, hi)

# e.g. accuracy: bootstrap_ci(correct_flags, np.mean)
```

---

## 🏗️ System design / applied

**38. Continuous eval on production traffic + drift alerts.**
**Sample** a privacy-safe slice of live traffic; run **automatic evals** (LLM-judge faithfulness/quality, safety classifiers, latency/cost) on a schedule; track metrics over time with **control charts / drift detection** (compare distributions vs a baseline window). **Alert** on statistically significant degradation. Periodically route a sample to **human review** to validate judges. Maintain a labeled golden set for regression and dashboards by segment (locale, intent).

**39. Eval harness a 50-person team can extend safely.**
- **Plugin/registry architecture:** tasks, datasets, metrics, and judges are self-contained modules registered by name — add new ones without touching the core.
- **Standard interfaces** (dataset → model → metric) and config-driven runs for reproducibility.
- **Isolation:** versioned datasets, pinned model/prompt versions, deterministic seeds; results stored with metadata.
- **CI integration:** shared regression suite runs on every model change; ownership/CODEOWNERS per task to avoid collisions. Caching to avoid recompute.

**40. Trustworthy internal leaderboard.**
- **Datasets:** a mix of capability, safety, robustness, and product-representative tasks; **private held-out** + rotating fresh sets; documented provenance.
- **Judges:** validated LLM-judges (order-swap, length-control, ensemble) **calibrated against human labels**, plus periodic human eval.
- **Contamination policy:** scan all candidate models' training data against eval sets where possible; canaries; time-split data.
- **Stats:** report CIs and significance, not raw ranks.
- **Refresh cadence:** rotate/refresh evals regularly; freeze a version per reporting period; publish methodology so results are auditable.

---

## 🐞 Debugging

**41. Same model, different scores on "equivalent" prompts.**
**Prompt-format sensitivity** — LLMs are brittle to phrasing, option order, delimiters, and few-shot examples. Don't pick the flattering format: **standardize** prompts, **report across several formats** (mean ± variance), test few-shot stability, and treat large swings as a robustness finding to fix (prompt normalization, more robust training).

**42. Judge always prefers the longer answer.**
**Verbosity bias.** Mitigate by: instructing the judge to ignore length and judge correctness/relevance, **controlling for length** (truncate/match, or include length as a covariate), normalizing scores by length, or using pairwise pairs matched on length. Validate the fix against human labels.

**43. Suspiciously high scores right after a data refresh.**
First hypothesis: **contamination** — the refresh pulled benchmark/eval data (or near-duplicates) into training. Run n-gram/MinHash overlap scans against eval sets and compare to perturbed/fresh equivalents before trusting the gains.

**44. A/B shows no difference but you "know" it's better.**
Likely an **underpowered test** (too few users/events to detect the effect), **wrong/insensitive metric** (measuring something the change doesn't move), **segment/heterogeneous effects** (better for some users, worse for others, averaging out), or **confounds** (novelty effect, latency regression offsetting quality). Fix: power analysis + larger/longer test, a metric aligned to the improvement, segment analysis, and check latency/guardrail metrics.

---

## What strong answers share
Treating evaluation as **experiment design with uncertainty** (CIs, power, significance); **intelligent distrust** of benchmarks (contamination, Goodhart, format sensitivity); ability to **separate sub-system failures** (retrieval vs generation); and validating cheap proxies (**judges**) against human ground truth.

---
Back to [questions](interview-questions.md) · [Stage README](README.md) · [Index](../README.md)
