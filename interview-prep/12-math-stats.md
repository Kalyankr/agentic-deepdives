# 12 · Math, Statistics & Classic ML

> Not every round is about transformers. Phone screens and ML-fundamentals rounds still probe
> **probability, statistics, the math behind training, and pre-LLM ML** — and these power the *right*
> answers elsewhere (why you put a CI on an eval, why cross-entropy is the loss, why a metric is wrong
> for imbalanced data). Lead with intuition, then the formula.

---

## Probability & statistics

- **Q: Bayes' rule, in words and symbols?**
  $P(A\mid B) = \dfrac{P(B\mid A)\,P(A)}{P(B)}$ — posterior ∝ likelihood × prior. It's how you update
  beliefs with evidence (spam filters, diagnostics, the base-rate fallacy).

- **Q: Expectation and variance of a sum?**
  $E[aX+bY]=aE[X]+bE[Y]$ always; $\mathrm{Var}(X+Y)=\mathrm{Var}(X)+\mathrm{Var}(Y)$ **only if
  independent** (else add `2·Cov`). Variance has squared units; std dev shares the data's units.

- **Q: Law of large numbers vs central limit theorem?** LLN: the sample mean converges to the true
  mean. CLT: the *sample mean's distribution* approaches Normal with **SE = σ/√n**, regardless of the
  population shape — the basis of confidence intervals.

- **Q: Standard error of a proportion (e.g. accuracy `p` on `n` items)?**
  $\mathrm{SE}=\sqrt{p(1-p)/n}$. To halve the SE you need **4× the data**. This is why small eval sets
  give noisy scores.

- **Q: Confidence interval intuition?** A 95% CI ≈ `estimate ± 1.96·SE`. It means: over many repeats,
  ~95% of such intervals contain the true value (not "95% probability the truth is here").

- **Q: When use the bootstrap?** When you can't write the SE in closed form (medians, BLEU, win-rates,
  ranking metrics): resample with replacement many times, recompute the metric, take percentiles. The
  go-to for **eval uncertainty**.

- **Q: p-value and significance?** The probability of a result this extreme **if the null were true**.
  `p < 0.05` ≠ "important" — pair it with an **effect size** and a CI. Beware **p-hacking** and the
  **multiple-comparisons** trap (testing 20 things → expect ~1 false positive at α=0.05; correct with
  Bonferroni/FDR).

- **Q: Designing an A/B test (ties to online evals)?** Define one **primary metric** + guardrails;
  compute **sample size** from the minimum detectable effect, baseline rate, α, and **power** (1−β,
  usually 0.8); randomize properly; don't peek/stop early without sequential corrections; check for
  novelty effects.

- **Q: Type I vs Type II error?** Type I = false positive (shipped a non-improvement). Type II = false
  negative (missed a real win). Power = 1 − P(Type II).

---

## The math behind training

- **Q: Why is cross-entropy the loss for classification/LMs?** It's the **negative log-likelihood**
  under a categorical model: minimizing CE = maximizing the probability of the data (MLE). It also
  equals `H(p) + KL(p‖q)`, so minimizing CE minimizes the **KL divergence** from data to model.

- **Q: KL divergence — definition and properties?**
  $D_{KL}(P\|Q)=\sum_x P(x)\log\frac{P(x)}{Q(x)}$. Non-negative, **asymmetric** (not a distance), zero
  iff `P=Q`. Shows up in VAEs, PPO's penalty, and distillation.

- **Q: Entropy and perplexity?** Entropy = expected surprise `−Σ p log p`. For LMs,
  `perplexity = exp(cross_entropy)` — the effective branching factor. Random-init loss ≈ `ln(vocab)`.

- **Q: MLE vs MAP?** MLE maximizes likelihood; MAP adds a **prior** (≡ regularization). L2 regularization
  = a Gaussian prior; L1 = a Laplace prior.

- **Q: Gradient descent family?** SGD (noisy, cheap), +**momentum** (accelerate consistent directions),
  **Adam/AdamW** (per-parameter adaptive via 1st/2nd moments; AdamW **decouples** weight decay). LR is
  the single most important hyperparameter; use **warmup + decay**.

- **Q: Vanishing/exploding gradients — causes and fixes?** Deep products of Jacobians shrink/blow up.
  Fixes: residual connections, normalization, good init, gradient clipping, non-saturating activations.

- **Q: Why does normalization help?** Stabilizes activation/gradient scale across layers, smooths the
  loss landscape → higher learnable LR. BatchNorm (uses batch stats; train/eval differ) vs LayerNorm/
  RMSNorm (per-example; the Transformer choice).

---

## Bias–variance & generalization

- **Q: Bias–variance trade-off?** Test error ≈ bias² + variance + irreducible noise. **High bias** =
  underfit (too simple); **high variance** = overfit (too sensitive to the training set). More
  data/regularization cuts variance; a richer model cuts bias.

- **Q: How do you fight overfitting?** More/augmented data, L1/L2, dropout, early stopping, smaller
  model, ensembling, cross-validation for honest estimates.

- **Q: L1 vs L2 regularization?** L2 (ridge) shrinks weights smoothly; L1 (lasso) drives some to exactly
  zero → **sparsity/feature selection**. (Connects to LoRA's low-rank prior on the *update*.)

- **Q: Why cross-validation, and when not?** k-fold gives a lower-variance performance estimate on small
  data. Avoid naive k-fold with **temporal** data (use forward-chaining) or **grouped** data (split by
  group to prevent leakage).

---

## Evaluation metrics (know when each is wrong)

- **Q: Precision vs recall vs F1?** Precision = of predicted-positive, how many are right
  (`TP/(TP+FP)`); recall = of actual-positive, how many you caught (`TP/(TP+FN)`); F1 = their harmonic
  mean. Pick by cost: recall for cancer screening, precision for spam.

- **Q: Why is accuracy misleading?** Under **class imbalance** (99% negatives), predicting "negative"
  scores 99%. Use precision/recall, PR-AUC, or balanced accuracy.

- **Q: ROC-AUC vs PR-AUC?** ROC-AUC = P(score(positive) > score(negative)), threshold-free; can look
  optimistic under heavy imbalance. **PR-AUC** focuses on the positive class and is more informative
  when positives are rare.

- **Q: What is calibration (ECE)?** Whether predicted confidence matches empirical accuracy (of things
  predicted at 0.8, ~80% should be correct). Important when probabilities drive decisions; fix with
  temperature scaling.

- **Q: Regression metrics?** MAE (robust, same units), MSE/RMSE (penalizes large errors), R² (variance
  explained). Choose by how much you care about outliers.

---

## Classic ML (one-liners that still come up)

- **Logistic regression:** linear model + sigmoid; trained with cross-entropy; interpretable, strong
  baseline.
- **Decision trees / Random Forests:** axis-aligned splits; forests = bagged de-correlated trees (cut
  variance).
- **Gradient-boosted trees (XGBoost/LightGBM):** sequential residual fitting; **still SOTA on most
  tabular data** — say this when someone reaches for a neural net on a spreadsheet.
- **SVM:** max-margin separator; kernels for non-linearity.
- **k-NN:** lazy, distance-based; suffers in high dimensions (curse of dimensionality).
- **k-means / GMM:** unsupervised clustering (hard vs soft assignments). You built cosine k-means in
  [lab06](../labs/lab06_rag/).
- **PCA / SVD:** linear dimensionality reduction by top eigen/singular directions — the same low-rank
  idea behind LoRA and embedding compression.
- **Bagging vs boosting:** bagging reduces **variance** (parallel, independent); boosting reduces
  **bias** (sequential, focuses on errors).

---

## Linear algebra & info theory you should reach for

- **Matrix shapes** are the #1 debugging tool — narrate them. A `(B,T,d)` activation × `(d,d)` weight → `(B,T,d)`.
- **Eigen/SVD:** any matrix `= UΣVᵀ`; truncating Σ gives the best low-rank approximation (Eckart–Young)
  — the math licence for LoRA, PCA, PQ.
- **Dot product** = unnormalized similarity; **cosine** removes magnitude (the retrieval metric).
- **Entropy/CE/KL** (above) are the currency of losses, distillation, and RLHF penalties.

> Interview reflex: when given any metric or method, **name the assumption it makes and the case where
> it breaks** ("AUC under 99% imbalance looks great but PR-AUC tells the real story"). That's the senior
> signal — and it ties straight back to [04-applied-llm](04-applied-llm.md) evals and the
> [CI-eval notebook](../notebooks/README.md).
