# 🧬 LLM Applications in Life Sciences & Biochemistry (Real-Time Focus)

> Reference notes on where LLMs / sequence-foundation-models fit in biochemistry, drug discovery, genomics, and adjacent fields — with an emphasis on **real-time** applications. Captured for future reference; a candidate seed for a future system-design module or capstone.
>
> 📚 Part of the [LLM Mastery course](README.md). Builds directly on the [agentic coding CLI](system-design/claude-code-cli/README.md) (the agentic loop), the [RAG platform](system-design/rag-platform/README.md), and the [LLM inference service](system-design/llm-inference/README.md).

---

## 1. The mental model

LLMs show up in life sciences in **three primitives** — all of which you already have designs for:

1. **Biology *is* a language.** Proteins (amino acids), DNA/RNA (nucleotides), and small molecules (SMILES/SELFIES strings) are token sequences, so transformer "language models" trained on them work remarkably well (ESM for proteins, Evo for DNA, ChemBERTa for molecules). Same architecture, different vocabulary.
2. **LLM-as-orchestrator.** A general LLM in a **think → act → observe** loop calling *specialized scientific tools* (docking, BLAST, AlphaFold, a lab robot). This is exactly the [Claude Code agent](system-design/claude-code-cli/README.md) — swap `grep`/`edit` for wet/dry-lab tools (see **ChemCrow**, **Coscientist**).
3. **Literature / knowledge RAG.** Grounded retrieval over PubMed, patents, internal assay data — your [RAG platform](system-design/rag-platform/README.md) verbatim.

### Three flavors of "real-time"
The useful axis to design around is **what kind of real-time** the application needs:

| Flavor | Latency | What it looks like |
|---|---|---|
| **Interactive copilot** | ms–seconds | a scientist iterates with an agent at the bench/desk |
| **Streaming instrument data** | ms–seconds | the model consumes a live sensor/sequencer feed |
| **Closed-loop autonomous lab** | minutes–days | agent designs → robot runs → analyzes → redesigns continuously |

---

## 2. Application catalog (by real-time flavor)

### A. Interactive design copilots — *the agentic loop, bio edition*
- **Protein / binder design copilot.** Describe a target → agent proposes sequences (ESM-3 / ProteinMPNN) → folds & scores them (AlphaFold3 / ESMFold) → checks stability/binding → iterates. Real-time = the **design ↔ predict** loop the scientist watches and steers.
- **Drug-molecule copilot.** Generate candidate molecules (SMILES) → predict ADMET / toxicity / binding affinity → run **retrosynthesis** ("can we even make it?") → filter. This is **ChemCrow's** pattern: an LLM augmented with ~18 expert chemistry tools.
- **Bioinformatics copilot.** Natural language → executable pipeline (alignment, variant calling, differential expression). A clean RAG-over-tools + code-gen problem.
- **Antibody / enzyme engineering assistant.** Suggest mutations, predict ΔΔG (stability), rank variants for the next round.

### B. Streaming instrument analysis — *genuinely real-time ML*
- **Adaptive nanopore sequencing ("Read Until").** Basecall reads *as the DNA translocates the pore* and **eject** uninteresting molecules in milliseconds to enrich targets. A flagship real-time bio-ML system (selective sequencing).
- **Real-time pathogen / outbreak ID.** Stream metagenomic reads during an outbreak → classify organism + antimicrobial-resistance genes live.
- **Mass spec / cryo-EM / microscopy.** Live peptide identification or particle picking as data streams off the instrument.
- **Wearable / continuous biosensors.** Continuous glucose or metabolite streams → on-device inference + LLM-generated explanation/alerts.

### C. Closed-loop self-driving labs — *the agentic CLI, but it runs experiments*
- **Autonomous experimentation.** A planner LLM proposes an experiment → robotic lab executes → instruments return data → agent analyzes and proposes the next experiment, looping for hours/days. **Coscientist** (GPT-4 driving a chemistry lab) and **A-Lab** (Berkeley, autonomous materials synthesis) are real, published systems.
- **Design–Build–Test–Learn (DBTL) for synthetic biology.** LLM reasoning + Bayesian optimization over the experiment space.
- **The hard part is safety** — this is your **permissions + sandboxing** subsystem, except the blast radius is a real reactor/reagent. Human-in-the-loop gating on dangerous steps is mandatory.

### D. Clinical / point-of-care real-time
- **Live variant interpretation** at the point of care; rapid genetic diagnosis from sequencing.
- **Deterioration / sepsis prediction** from streaming EHR + vitals, with an LLM generating the rationale a clinician can audit.
- **Real-time decision support** grounded (RAG) in guidelines to control hallucination.

---

## 3. Foundation models you'd build on

| Domain | Models (examples) | Use |
|---|---|---|
| **Proteins (sequence)** | ESM-2, **ESM-3** (generative), ProtTrans | embeddings, function prediction, design |
| **Protein structure** | **AlphaFold3**, ESMFold, RoseTTAFold, Boltz-1 | sequence → 3D structure / complexes |
| **Protein design** | **RFdiffusion**, ProteinMPNN | de novo backbones + sequences |
| **DNA / RNA** | **Evo**, Nucleotide Transformer, DNABERT-2, HyenaDNA | variant effects, regulatory elements, generation |
| **Single-cell** | **scGPT**, Geneformer, scFoundation | cell-type, perturbation response |
| **Small molecules** | ChemBERTa, MolFormer, MolT5 | property prediction, captioning |
| **Reactions / synthesis** | Molecular Transformer, IBM RXN | forward reaction & retrosynthesis |
| **Agents (LLM + tools)** | **ChemCrow**, **Coscientist** | orchestrate the above as tools |

> Note the parallel to your course: these are just transformers with a **domain tokenizer** + **domain pretraining corpus** (Stage 1–2), often **fine-tuned/adapted** (Stage 3), **served** behind an inference API (Stage 5), and wrapped in **RAG or an agent** (Stage 6 / system-design).

---

## 4. Real systems worth studying
- **AlphaFold2/3** (DeepMind) — protein structure; redefined structural biology.
- **ESM-2 / ESM-3** (Meta / EvolutionaryScale) — protein language models; ESM-3 is generative ("programming biology").
- **RFdiffusion + ProteinMPNN** (Baker lab) — de novo protein design.
- **ChemCrow** (Bran et al., 2023) — GPT-4 + ~18 chemistry tools; the cleanest "agentic CLI for chemistry."
- **Coscientist** (Boiko et al., *Nature* 2023) — LLM autonomously plans & executes chemical experiments.
- **A-Lab** (Berkeley, *Nature* 2023) — autonomous robotic lab synthesizing novel materials.
- **Evo** (Arc Institute) — DNA foundation model across molecular → genome scale.
- **Nanopore "Read Until" / adaptive sampling** — real-time selective sequencing.

---

## 5. Most buildable as a prototype right now
1. **Protein-design copilot** — ESM-2 / ESMFold are open and runnable; a clean agentic loop (propose → fold → score → iterate).
2. **ChemCrow-style chemistry agent** — LLM + RDKit + a retrosynthesis API + property predictors; almost a **direct re-skin of the Claude Code design**.
3. **Literature hypothesis-generation RAG** over bioRxiv / PubMed — reuses the [RAG platform](system-design/rag-platform/README.md) wholesale; lowest barrier, immediately useful.

---

## 6. System-design challenges (why this is good interview/portfolio material)
- **Latency vs. heavy tools.** Folding/docking can take seconds–minutes → async jobs, queues, streaming partial results (ties to the [inference service](system-design/llm-inference/README.md) + [Claude Code](system-design/claude-code-cli/README.md) async-tool patterns).
- **Grounding & hallucination.** Science punishes confident wrong answers → RAG + verification-in-the-loop (run the predictor, don't trust the prose).
- **Tool orchestration & cost.** Many specialized models/tools → routing, caching, budget caps.
- **Safety & permissions.** Autonomous labs need irreversible-action gating, sandboxing, and human approval — plus **dual-use/biosecurity** screening (don't help design hazards). Ties to [Stage 8 — Safety & Security](stage-8-safety-security/README.md).
- **Data: point-in-time correctness.** Assay results, patient data → your [feature store](system-design/feature-store/README.md) concerns (train/serve skew, leakage).

---

## 7. Caveats — what makes science hard for LLMs
- **Ground truth is wet-lab, slow, and expensive** — predictions are hypotheses until validated.
- **Hallucination is unacceptable** in clinical/safety contexts → always ground + cite + verify.
- **Distribution shift** — models extrapolate poorly to truly novel chemistry/biology.
- **Regulatory / privacy** — HIPAA/GDPR for patient data; validation burden for anything clinical.
- **Biosecurity / dual-use** — generative bio + chemistry models need misuse safeguards.

---

## 8. Next-step ideas (pick later)
- Promote one of these into a full **system-design module #8** (e.g., "🧬 Self-driving lab / bio research agent" or "Protein-design copilot") in the HLD + 42-Q + answers + cheat-sheet format.
- Or scaffold a **runnable prototype** (ChemCrow-style chemistry agent, or an ESMFold protein-design loop).
- Or add a **capstone** under [capstones](capstones/README.md): "LLM agent for retrosynthesis planning" or "RAG hypothesis generator over bioRxiv."

---

[← Back to course index](README.md) · Related: [Claude Code CLI](system-design/claude-code-cli/README.md) · [RAG platform](system-design/rag-platform/README.md) · [LLM inference service](system-design/llm-inference/README.md) · [Stage 8 — Safety & Security](stage-8-safety-security/README.md)
