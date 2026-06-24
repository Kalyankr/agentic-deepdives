# 09b — Full Spoken Answers (say-it-in-the-interview)

> Companion to [09-interview-questions.md](09-interview-questions.md). Every question answered **the way you'd actually say it out loud** — first person, structured, complete but not rambling. Aim for 45–90 seconds each. Phrases in *(italics)* are optional asides to go deeper if the interviewer leans in.
>
> 🔬 For the **deeper follow-up probes** an interviewer asks after these answers, see [09c — Follow-Up Questions](09c-followup-questions.md).

**How to deliver these:** open with a one-line definition, give the structure/why, add one concrete example, then stop. Don't monologue — pause and let them follow up.

---

## A. Fundamentals & concepts

### 1. What is an AI agent?
"The way I define it: an AI agent is a system that uses an LLM as its reasoning engine to decide its *own* control flow toward a goal. Instead of me hardcoding the steps, the model looks at the goal and the current state, decides the next action — usually a tool call — observes the result, and loops until it's done or hits a stop condition. My mental model is: **an agent equals an LLM, plus a loop, plus tools, plus memory, plus a goal and a stop condition.** The LLM is the brain, tools are the hands, memory is the state. The key word is *dynamic* — the model decides the path, not my code. A coding assistant that reads files, runs tests, and fixes errors on its own is a good example."

### 2. Agent vs. workflow vs. single LLM call?
"They sit on a spectrum of who controls the flow. A **single LLM call** has no control flow — one input, one output, like 'summarize this.' A **workflow** has control flow but *I* define it in code — a fixed path like retrieve, then stuff, then generate in a RAG pipeline; the LLM fills in steps but doesn't choose them. An **agent** is where the *LLM* decides the control flow at runtime — which tool, in what order, and when to stop. Anthropic frames it well: workflows orchestrate LLMs through predefined code paths, agents let the LLM dynamically direct its own process. My rule of thumb is to use the simplest one that works — agents are the most powerful but also the least predictable and hardest to evaluate."

### 3. Name the five components of an agent.
"Five pieces. One, the **model or reasoning core** — the LLM that interprets state and picks the next action. Two, **tools or actions** — functions, APIs, code execution, retrieval, even other agents; that's how it affects the world. Three, **memory** — short-term in the context window plus long-term in external stores so it's stateful. Four, **planning and orchestration** — how goals get decomposed and steps sequenced, which can be implicit in the model or explicit in code. And five, the **control loop and policy** — the loop itself, stop conditions, guardrails, and any human-in-the-loop checkpoints. If you take away the loop and tools, you're just back to a chatbot."

### 4. Draw the agent loop.
"It's a five-step cycle. **Perceive** — read the goal, current state, and the latest observation. **Reason** — think about what to do next; in ReAct that's an explicit 'thought.' **Act** — call a tool or produce an action. **Observe** — get the result back from the environment. And **Check** — am I done, did I error, did I blow the budget? If not, loop back to reason. *The thing I always emphasize is that the check step is mandatory — every agent needs hard stop conditions or it'll loop forever and burn money.*"

### 5. What are mandatory stop conditions?
"At minimum: a **max-steps** cap, a **token-or-cost budget**, and a **wall-clock timeout**. On top of that I add **repeated-action detection** — if the agent calls the same tool with the same args twice, that's usually a loop, so I break it — and an explicit **terminate or finish tool** so the model can cleanly signal it's done. Unbounded loops are the number-one cause of runaway cost in production, so I treat stop conditions as non-negotiable, not an afterthought."

### 6. Levels of agency?
"I think of it as a ladder. Level zero is a **pure LLM call**. Level one is **LLM plus tools** — a router or function-calling chatbot where code drives but the model picks within a step. Level two is a **chained workflow** — a fixed DAG like prompt chaining or RAG. Level three is a **single ReAct agent** looping over tools. Level four is **multi-agent** — an orchestrator with specialists. And level five is **fully autonomous** — the agent sets its own sub-goals and maybe spawns other agents. As you go up, you gain capability but lose predictability and gain cost and eval difficulty. The senior instinct is to pick the *lowest* level that solves the problem."

### 7. When would you NOT use an agent?
"Several cases. If a **single prompt or a fixed workflow** already solves it — that's cheaper, faster, and testable, so adding a loop is just risk. If **latency or cost budgets are tight**, because agents make multiple model calls. If **errors are costly and hard to reverse** and I can't put strong guardrails or a human in the loop. And if I **can't evaluate or observe** what it's doing — an agent I can't measure is a liability. My honest answer in interviews is that most 'make it agentic' requests are better served by a router or a chain; complexity is a cost you pay in latency, money, and debuggability, so I only spend it when dynamic tool selection is genuinely required."

### 8. What changed to make agents viable now?
"A few things converged. **Tool and function calling** became reliable — models now emit valid structured calls, even in parallel. **Long context** windows, hundreds of thousands to millions of tokens, make multi-step state feasible. **Reasoning got better**, especially with RL-trained reasoning models, so planning actually works. **Standard protocols** like MCP for tools and A2A for agent-to-agent cut integration cost. And **inference got cheap enough** that running a multi-call loop is economically sane. None of these alone was enough; it's the combination that tipped agents from demo to production."

### 9. What is scaffolding?
"Scaffolding is all the engineering *around* the model that turns a raw LLM into an agent — the loop, the prompt templates, the output parsing, the tool definitions and execution, the memory management, and the guardrails. The model provides the reasoning, but the scaffolding provides the structure and safety. *A point I like to make: a lot of 'agent quality' actually comes from good scaffolding — clear tools and solid error handling — not from a cleverer prompt.*"

### 10. Define grounding.
"Grounding is connecting the model to real, external data or state so its outputs are factual instead of coming purely from its parametric memory. RAG is the classic example — I retrieve relevant documents and put them in context so the answer is based on actual sources, ideally with citations. Tool results ground the model too. Ungrounded models hallucinate confidently; grounding is how you tie them to ground truth."

### 11. Is more autonomy always better?
"No, and that's a trap question. More autonomy buys capability but costs you **predictability, evaluability, and safety**. A highly autonomous agent can handle open-ended tasks, but it's harder to test, harder to debug when it fails, more expensive, and a bigger security surface. The right amount of autonomy is tied to the cost and reversibility of mistakes — for low-risk, reversible tasks I'll allow more; for irreversible or high-stakes actions I keep a human in the loop and constrain it tightly. So I scale autonomy to the risk, not to the hype."

### 12. What's context engineering?
"Context engineering is the discipline of curating exactly what goes into the model's context window at each step — the instructions, the relevant memory, the available tools, the retrieved data, and the current state — all within a finite token budget. It's broader than prompt engineering, which is mostly about wording. The hard part is *selection and timing*: getting the right information in at the right moment without overloading the window. *I usually mention the failure modes I'm guarding against — poisoning, distraction, confusion, and clash — and the 'lost in the middle' effect, which is why I put the most important content at the start or end.*"

---

## B. Reasoning & planning

### 13. Explain Chain of Thought.
"Chain of Thought is prompting the model to produce intermediate reasoning steps before the final answer — literally 'let's think step by step.' It works because it lets the model allocate more compute to multi-step problems and externalize its intermediate state instead of jumping to a conclusion. There are variants: zero-shot, few-shot with worked examples, and self-consistency where you sample several chains and majority-vote. The caveat I always add is that CoT is reasoning *without* feedback — the model can produce a fluent but completely wrong chain — which is exactly why agents interleave reasoning with real actions."

### 14. Self-consistency?
"Self-consistency is a reliability trick: instead of taking one reasoning chain, you sample N of them at some temperature and then take the majority-vote answer. The intuition is that there are many paths to the right answer but wrong answers tend to be inconsistent, so voting filters out the noise. It's a cheap way to boost accuracy on reasoning tasks — the cost is N times the tokens, so I use it when accuracy matters more than latency, and I'd cap N."

### 15. Explain ReAct and why it beats CoT for agents.
"ReAct stands for Reason plus Act. It interleaves the two: the model produces a thought, takes an action like a tool call, gets an observation back, then thinks again with that new information — thought, action, observation, repeat. The reason it beats plain CoT for agents is **grounding**. In CoT the model reasons from its priors alone, so it can hallucinate a whole chain. In ReAct every reasoning step is conditioned on real observations from the environment, so it hallucinates less, it can *recover* from errors using the feedback, and the trace is interpretable and debuggable. That's why ReAct is the canonical agent loop. The failure modes to watch are looping on a failing tool and bad output formatting, which I handle with loop detection and native function calling."

### 16. Reflexion / reflection?
"Reflection is when the agent critiques its own output and revises it. Reflexion, the paper, takes it further: the agent gets a feedback signal — tests failed, an error message, an evaluator score — generates a *verbal* self-reflection like 'I assumed one-indexing but the API is zero-indexed,' stores that reflection in episodic memory, and uses it on the next attempt. It's effectively reinforcement learning in natural language, within a single session, with no weight updates. I reach for it when there's a clear success signal and iteration actually helps — coding, math, structured generation. I avoid it when there's no reliable signal, because then you're just reflecting on noise, and I always cap the number of rounds because each one multiplies cost."

### 17. ReAct vs. Plan-and-Execute?
"The core difference is *when* the model decides. **ReAct** re-reasons at every single step — very adaptive, recovers well, but it pays for a reasoning call on each step and can drift on long tasks. **Plan-and-Execute** plans the whole thing up front, then an executor carries out the steps, optionally with a replanner if reality diverges. That's more efficient for long, decomposable tasks because you decide the path once instead of every step, the plan is auditable, and independent steps can be parallelized — but it's more rigid and a bad plan cascades. So: unpredictable, exploratory work leans ReAct; structured, decomposable work leans Plan-and-Execute with replanning."

### 18. Tree of Thoughts?
"Tree of Thoughts generalizes Chain of Thought from a single linear chain to a *tree* of reasoning branches. The model generates multiple candidate thoughts at each step, evaluates the intermediate states, and searches — BFS, DFS, or beam — toward the best solution, with the ability to backtrack. It's powerful for problems that need exploration and look-ahead, like puzzles or planning. The catch is cost: it's many model calls per node, so it's usually overkill for production agents. I bring it up mostly to show I understand the reasoning-versus-compute spectrum, and I'd mention LATS as the variant that combines tree search with reflection and tool use."

### 19. ReWOO?
"ReWOO — Reasoning Without Observation — is a token-efficiency optimization over ReAct. Instead of interleaving a reasoning call with every observation, it plans *all* the tool calls up front in one pass, then executes them, then does a final solve. By decoupling reasoning from observations you make far fewer LLM calls, which saves tokens and latency. The trade-off is less adaptivity — if an early result should change the plan, plain ReWOO won't notice without a replanning step. It's a nice answer when someone asks how you'd cut the cost of a ReAct agent."

### 20. How does prompting change with reasoning models (o-series/R1)?
"With reasoning models, you do *less* hand-holding. These models are RL-trained to produce long internal chains of thought before answering, so the reasoning I used to engineer into the prompt now lives inside the model. Practically that means I give them the **goal, the constraints, and the tools**, and let them plan — rather than spelling out step-by-step CoT instructions, which can actually hurt them. So my prompts get more declarative and outcome-focused, I lean less on few-shot reasoning examples, and I spend my effort on clearly specifying *what* success looks like instead of *how* to think."

### 21. Evaluator-optimizer pattern?
"It's a two-role loop: one LLM **generates** a candidate, a second LLM **evaluates** it against a rubric and gives concrete feedback, and the generator revises — repeat until the evaluator approves or you hit a budget. It's the workflow version of reflection. I use it when I have clear evaluation criteria and iteration measurably improves quality — things like translation, code, or structured writing. The two guardrails are a meaningful rubric, so the critic isn't vague, and a cap on rounds so it converges."

### 22. When is reflection NOT worth it?
"When there's no reliable success signal to reflect on — if I can't tell whether an attempt was good, reflection just amplifies the model's own biases and noise. Also when latency or cost is tight, since each reflection round can double or triple the calls. And on simple tasks where the first answer is almost always right, reflection adds cost for no gain. So my checklist is: is there a real signal, does iteration actually help, and can I afford the rounds — if not, I skip it or cap it hard."

### 23. Decompose a reasoning-strategy decision.
"I run a quick decision tree. Single, well-defined task with no tools — plain Chain of Thought, and self-consistency if accuracy is critical. Needs external info or actions and the path is unknown — ReAct. Long, decomposable task with clear sub-steps — Plan-and-Execute with replanning. Whenever there's a clear pass-fail signal and iteration helps — layer in reflection on top. Needs real exploration and backtracking and I can afford it — Tree of Thoughts or LATS. And if I'm on a reasoning model, I strip the scaffolding back and just give it the goal. The meta-principle is the same as everywhere in agents: use the lightest technique that hits the quality bar."

---

## C. Memory & context

### 24. Short-term vs. long-term memory?
"Short-term memory is what lives in the **context window** for the current task — the recent turns, the scratchpad, the current plan, the latest tool outputs. It's fast but bounded by the token limit. Long-term memory is **external and persistent** — vector stores, SQL, key-value, a knowledge graph — holding things across sessions like the user profile, past conversations, or a knowledge base. The management problem differs: short-term is about *fitting* within the budget — windowing, summarization, eviction — while long-term is about *retrieval quality* — getting the right thing back when you need it. Together they turn a stateless model into a stateful agent."

### 25. Episodic vs. semantic vs. procedural memory?
"This is the cognitive analogy, and it maps cleanly. **Episodic** memory is specific past experiences — 'last time this user preferred bullet summaries,' or stored past trajectories; Reflexion's self-reflections are episodic. **Semantic** memory is facts and knowledge — domain facts, the user profile, entities — usually a vector store or knowledge graph. **Procedural** memory is *how to do things* — the system prompt, tool-use skills, learned routines; it often lives in code or a skill library. Quick examples: a user-profile store is semantic, a log of past episodes is episodic, and a reusable prompt or skill is procedural."

### 26. Manage an overflowing context window?
"I layer several techniques. **Rolling summarization or compaction** — periodically compress old turns into a running summary so I keep the gist without the tokens. A **sliding window** for raw recent turns. **Retrieve, don't stuff** — pull only the context relevant to the current step instead of carrying everything. **Offload large artifacts** to external memory and pass a handle or ID instead of the full blob. And in multi-agent setups, **split the work across sub-agents** with isolated contexts. Underneath all of it I do token-budget accounting so eviction is deliberate, and I pin the critical instructions at the start or end because of the lost-in-the-middle effect."

### 27. Walk through a RAG pipeline.
"Two phases. **Ingestion**, offline: take the documents, chunk them, embed each chunk, and store the vectors in an index with metadata. **Query**, at runtime: embed the user's query, do a similarity search to get the top-k chunks, optionally re-rank them with a cross-encoder for precision, stuff the best ones into the prompt, and have the model generate an answer grounded in them — ideally with citations back to the sources. The whole point is grounding the model in current or private data it wasn't trained on. *Where I'd go deeper is the quality knobs — chunking, hybrid retrieval, reranking — and evaluating retrieval and generation separately.*"

### 28. RAG quality knobs?
"Starting from ingestion: **chunking** — size and overlap, semantic versus fixed; too big adds noise, too small loses context. **Embeddings** — model choice and domain fit. On retrieval: **dense versus sparse versus hybrid** — I usually combine vector search with BM25 because keywords still matter; **top-k** and **metadata filters**. Then **re-ranking** — a cross-encoder over the top-k to push the truly relevant chunks up. And **query transformation** — rewriting, decomposition, or HyDE to bridge the gap between how users ask and how documents are written. If RAG is underperforming, I diagnose *where* — retrieval or generation — before touching anything."

### 29. Agentic RAG vs. naive RAG?
"Naive RAG is a fixed pipeline: every query triggers exactly one retrieval, then generation. Agentic RAG makes retrieval a **tool the agent decides to use**. The agent decides *whether* it even needs to retrieve, *what* query to issue, *which* source or index to hit, and crucially it can **iterate** — retrieve, reason about what's missing, then retrieve again — and it can decompose a complex question into sub-queries. So it handles multi-hop and ambiguous questions far better than a single-shot pipeline. The cost is more model calls and more orchestration, so for simple lookups naive RAG is still fine."

### 30. 'Lost in the middle'?
"It's an empirical finding that LLMs attend most strongly to information at the **beginning and end** of a long context and can effectively miss things buried in the middle — performance is U-shaped with position. The practical implication for agents is placement matters: I put the most important instructions and the most relevant retrieved chunks at the start or the end of the window, not the middle, and I keep contexts lean rather than dumping everything in and assuming the model will find the needle."

### 31. Context failure modes?
"I name four. **Poisoning** — a hallucination or error gets written to memory and then reused as if it were true. **Distraction** — so much irrelevant context that it dilutes the model's attention. **Confusion** — conflicting or duplicated information in the window. And **clash** — newly retrieved info directly contradicts what's already there. The common thread is that *more* context isn't *better* context; my job in context engineering is to keep the window relevant, consistent, and clean, not just full."

### 32. How do you evaluate RAG?
"I evaluate retrieval and generation **separately**, because a bad answer could be either's fault. On the retrieval side: **context precision and recall** — did I fetch the relevant chunks and not junk. On the generation side: **faithfulness or groundedness** — is every claim supported by the retrieved context — and **answer relevance** — does it actually address the question. Frameworks like RAGAS operationalize these, often with LLM-as-judge. I build a small labeled eval set from real queries, track these metrics over changes, and that tells me whether to fix chunking and retrieval or fix the generation prompt."

### 33. When use a knowledge graph vs. vector DB?
"Vector databases are for **semantic similarity** — 'find me text that means roughly this' — and they're the backbone of most RAG and episodic memory. Knowledge graphs are for **entities and explicit relationships** — when the question requires multi-hop reasoning across connected facts, like 'which suppliers of my supplier are in this region.' That's where GraphRAG shines, because vector search alone struggles to traverse relationships. In practice I often combine them: vectors for fuzzy recall, a graph for structured, multi-hop connections."

---

## D. Tools & function calling

### 34. How does function calling work? Who executes?
"The flow has four steps. I send the model the messages plus a list of tool schemas — name, description, and JSON-schema parameters. The model decides to use a tool and returns a **structured tool call** — the function name and JSON arguments. The critical point: **the model does not execute anything; my code does.** My harness runs the actual function, gets the result, and passes it back as a tool message. Then the model reads that result and either calls another tool or gives the final answer. So the model only emits *intent*; the execution and therefore the security boundary live in my code. And because the arguments come back as schema-validated JSON, I get far fewer parsing errors than regexing free text."

### 35. Principles of good tool design.
"I treat tool design as more important than prompt cleverness. **Clear names and descriptions**, because the description is effectively a prompt the model uses to choose the tool — I say explicitly when to use it and when not to. **Narrow scope** — one tool, one job, not a god-tool with fifteen parameters. **Strong typed schemas** — enums over free strings, required versus optional, sane defaults, to constrain the input space. **Useful return values** — return what the model needs to decide the next step, plus a status and, on failure, an actionable error message. **Idempotency and safety** for anything with side effects, gating destructive actions behind confirmation. **Few, non-overlapping tools**, because overlap causes selection confusion. And **token-aware outputs** — paginate or summarize big results instead of dumping them."

### 36. Handle tool errors for recovery?
"The principle is: **errors are feedback the model can act on**, not stack traces. So I validate arguments before executing and return a clear validation message so the model can retry correctly — 'city not found, try a country code' beats a 500. For transient failures I do **timeouts and retries with backoff**. For anything with side effects I use **idempotency keys** so a retry doesn't double-charge. I add **circuit breakers and per-tool budgets** to stop runaway loops, and **graceful degradation** — a fallback tool or handing off to a human when something's truly down. The agent should adapt to a failing tool, not retry it blindly forever."

### 37. What is MCP and what problem does it solve?
"MCP, the Model Context Protocol, is an open standard from Anthropic that standardizes how applications expose tools, data, and prompts to LLMs. The analogy everyone uses is it's a USB-C port for AI — connect any MCP-compatible client to any MCP server without a bespoke integration. The problem it solves is the N-times-M explosion: if you have N apps and M tools, you used to write N times M custom integrations; MCP turns that into N plus M. You build a tool once as an MCP server and every MCP client can use it. The three primitives a server exposes are **tools** — model-callable actions, **resources** — readable data and context, and **prompts** — reusable templates. The one caveat I flag is that MCP servers are also a supply-chain and prompt-injection surface, so I vet them and scope their permissions."

### 38. MCP architecture?
"Three roles. The **host** is the AI application — Claude Desktop, an IDE, my agent. Inside the host sits one or more **clients**, each maintaining a one-to-one connection to a server. And the **server** exposes the actual capabilities — tools, resources, prompts. Communication goes over a couple of transports: stdio for local servers and HTTP with SSE for remote ones. So the host's client talks to a server, the server advertises what it can do, and the model calls those capabilities through that channel."

### 39. Computer-use agents — when and risks?
"Computer-use agents control a machine the way a human does — the model gets a screenshot, decides an action like click at these coordinates or type this, the action executes, a new screenshot comes back, and it loops. I'd use it when there's **no API** — legacy desktop apps, arbitrary websites, anything you can only drive through a GUI. The risks are real: it's slow, it's brittle to any UI change, it's expensive because of vision tokens, and it's a serious security exposure because it can do anything the logged-in user can. So I always sandbox it and put a human in the loop for sensitive actions."

### 40. Code-as-action (CodeAct)?
"CodeAct is a pattern where, instead of emitting a single JSON tool call per step, the agent writes **code** — usually Python — that calls tools, and that code runs in a sandbox. The advantage is expressiveness: in one step the agent can compose multiple tools, loop, branch, and manipulate data, which is awkward to express as a sequence of individual JSON calls. It's especially strong for data analysis and tasks that orchestrate several tools. The requirement is a secure sandbox — no network unless needed, least privilege — and capturing stdout and stderr so errors flow back as observations the agent can reflect on."

### 41. Agent keeps picking the wrong tool — fix?
"Almost always it's a tool-selection problem rooted in the descriptions or overlap, not the model being dumb. So first I **tighten each description** to say exactly when to use it and when not to. I **remove or merge overlapping tools** — if two are confusable, I consolidate them into one with a mode parameter. I **constrain the schema** with enums and required params so misuse is harder. I add a couple of **few-shot examples** of correct selection. And I **read the traces** to see whether the model is misreading the schema or misunderstanding the task — that tells me whether to fix the tool or the prompt."

### 42. Why not dump big tool outputs into context?
"Three reasons. **Cost** — every token in that dump is paid for on every subsequent step it stays in context. **Distraction** — a huge blob dilutes the model's attention and can trigger the lost-in-the-middle effect. And **overflow** — a few big outputs can blow the window. So instead I summarize or filter the output down to what the model needs to decide the next step, paginate large results, or store the full artifact externally and pass back a handle or ID. The tool should return *signal*, not a raw firehose."

### 43. Parallel tool calls — why useful?
"When the model needs several independent pieces of information or actions, it can request them all at once and my harness runs them concurrently instead of one after another. That cuts latency significantly for I/O-bound calls — say, fetching weather for three cities, or querying three data sources. The constraint is the calls have to be genuinely independent; anything where one result feeds the next still has to be sequential. It's a simple, high-leverage latency win."

---

## E. Multi-agent systems

### 44. When multi-agent vs. single agent?
"I decide based on **coupling and context pressure**, and I make sure to present both sides. A **single agent** wins when the work is tightly coupled — shared context prevents the fragmentation and conflicting-decision problems that wreck naive multi-agent setups; that's Cognition's argument from building Devin, and it's why coding agents are often single-threaded. **Multi-agent** wins when subtasks are genuinely independent and parallelizable, or each needs an isolated context to avoid overloading one window, or distinct specialization, tools, or models help — Anthropic's research system saw big gains exactly there. The thing I never forget to mention is the cost: multi-agent can burn many times more tokens, so the value has to justify it. So it's not a default — it's a tool for managing complexity and context."

### 45. Compare orchestrator-worker, hierarchical, network topologies.
"**Orchestrator-worker** is central control: a lead agent decomposes the task, delegates subtasks to workers — often in parallel — and synthesizes the results. It's the most common production pattern. **Hierarchical** is that same idea nested into a tree — supervisors of supervisors — for large, complex problems where mid-level managers coordinate sub-teams. **Network or peer-to-peer** is decentralized: any agent can hand off to any other, many-to-many. It's the most flexible but the hardest to control and the most prone to chaos, so in practice I constrain it with a routing policy. As complexity grows I tend to start central and only decentralize if I have a real reason."

### 46. Handoff/swarm pattern — where ideal?
"In a handoff or swarm pattern, control is *transferred* between agents — one agent is active and fully owns the conversation until it explicitly hands off to another. It's the model behind OpenAI's Swarm and Agents SDK. It's ideal for **routing-style problems**, classically customer support: a triage agent figures out intent and hands off to billing, or tech, or accounts, and that specialist owns the turn with its own tools and policy. It keeps each agent's context clean and its permissions scoped, and the handoff is explicit and easy to trace."

### 47. Group chat / debate?
"In a group chat, several agents share one conversation and a manager — or a round-robin or speaker-selection policy — decides who talks next. Debate is a variant where agents argue opposing positions and a judge or consensus resolves it. The benefit is that multiple perspectives and mutual critique can improve reasoning and factual accuracy — it's like a society of minds. The downside is it's token-heavy and can meander, so I use it for brainstorming, review, or hard reasoning problems where the quality lift is worth the cost, and I always cap the number of rounds."

### 48. Blackboard pattern?
"The blackboard pattern has agents read from and write to a shared workspace — the blackboard — and a controller decides which agent gets to act on the current state. Nobody talks directly; they collaborate *through* the shared state. It's good for opportunistic, loosely-coupled problem solving, where different specialists can contribute whenever the state is ripe for their expertise. It also gives you a single source of truth, which helps avoid the conflicting-information problem you get when agents only message each other."

### 49. How do agents communicate?
"Four main mechanisms. **Shared message history** — everyone sees everything, like a group chat; simple but noisy and token-heavy. **Direct messages or handoffs** — targeted transfer of control plus the context that matters. **Shared state or a blackboard** — agents read and write a common store, like a LangGraph state object. And **structured artifacts** — agents exchange typed objects, a plan or a JSON result, rather than raw chat. The biggest cost mistake I see is passing full transcripts everywhere; the fix is to pass *distilled artifacts* — each agent gets what it needs, not the entire history."

### 50. MCP vs. A2A?
"They operate at different layers. **MCP** standardizes how an agent talks to **tools and data** — the vertical connection down to capabilities. **A2A**, Google's Agent2Agent, standardizes how agents talk to **each other** — the horizontal connection across agents, even across vendors and frameworks. In A2A an agent publishes an Agent Card describing its capabilities, endpoint, and auth, and other agents discover it and delegate *tasks* to it over HTTP and JSON-RPC, treating it as an opaque peer with no shared memory or tools. The one-liner I use: **MCP is how an agent uses tools; A2A is how agents talk to each other** — together they're the USB-C plus networking of the agent ecosystem."

### 51. Failure modes of MAS?
"Several to name. **Cascading errors** — an early mistake poisons everything downstream. **Coordination failure or deadlock** — agents wait on each other or loop. **Context fragmentation** — agents lack information their siblings hold and make contradictory decisions; that's the core critique of naive multi-agent. **Cost explosion** from chatty agents. **Duplicate or conflicting work** when ownership is unclear. And **harder evaluation** — figuring out *which* agent caused a failure. My mitigations are clear role boundaries, structured handoffs instead of chat, an orchestrator that owns the plan, shared state as a single source of truth, per-agent budgets, and end-to-end tracing with per-agent attribution."

### 52. Roles you'd define?
"Common roles: an **orchestrator or planner** that decomposes and delegates; a **researcher or retriever** that gathers information; **workers or specialists** that do focused subtasks like coding, SQL, or analysis; a **critic or reviewer** that checks quality — the evaluator half of evaluator-optimizer; a **synthesizer or writer** that merges results into the final output; a **router or triage** agent that classifies and dispatches; and sometimes a **guardian or safety** agent that enforces policy. A pattern I call out a lot is generator-plus-critic, because separating production from review reliably improves quality."

### 53. Why can multi-agent use 15× more tokens?
"It compounds from a few sources. Each agent carries its own context and system prompt, so you're paying for context multiple times. Inter-agent communication adds chatter on top of the actual work. Parallel agents all consume tokens simultaneously. And orchestration — planning, delegating, synthesizing — is itself extra model calls. Anthropic reported their multi-agent research system used about fifteen times the tokens of a single chat. So the honest framing is that multi-agent trades tokens for capability and latency, and I only spend that when the task value justifies it — and I set per-agent budgets to keep it bounded."

### 54. How to localize a failure in MAS?
"The enabler is **observability with per-agent attribution** — I trace every agent's inputs, outputs, tool calls, and handoffs, so when the final result is wrong I can walk back through the trajectory and see which agent's step introduced the error. Passing **structured artifacts** rather than free-form chat makes this much easier, because I can inspect a typed object at each boundary. And **checkpoints** let me replay or resume from a known-good state. Without that tracing, debugging multi-agent failures is basically guesswork, which is exactly why I build it in from the start."

---

## F. Frameworks & protocols

### 55. LangGraph vs. AutoGen?
"They embody two different mental models. **LangGraph** represents the agent system as an explicit **stateful graph** — nodes are functions or agents, edges including conditional ones are the control flow, and a typed shared state flows through. It supports cycles, persistence and checkpointing, and human-in-the-loop interrupts, so it gives you maximum control and observability — that's my pick for complex, production, controllable flows. **AutoGen** models agents as **conversational** participants, often in a group chat where a manager picks the next speaker, with first-class code execution. That's great for collaborative, exploratory, code-writing multi-agent tasks. So: LangGraph when I need deterministic control and durability; AutoGen when natural agent-to-agent conversation and rapid experimentation matter more."

### 56. CrewAI model?
"CrewAI is the **role-based** framework. You define agents with a role, a goal, and a backstory, plus their tools; you define tasks with a description and expected output and an owner; and you pick a process — sequential or hierarchical — that runs the crew. It's opinionated and very readable — standing up a Researcher plus Writer plus Editor team takes very little code, which makes it great for getting going fast. The trade-off versus LangGraph is less low-level control, and the clean abstraction can hide failure modes, so for very complex or highly-controlled flows I'd lean elsewhere — though CrewAI's Flows add more control when you need it."

### 57. OpenAI Agents SDK (ex-Swarm)?
"It's a lightweight, production-oriented framework built on a few primitives: **agents** with instructions and tools, **handoffs** to transfer control between agents, **guardrails** for input and output validation, **sessions** for memory, and built-in tracing. It's the supported successor to the experimental Swarm. What it adds over a raw chat loop is the explicit handoff and guardrail abstractions plus tracing, without the ceremony of a heavier framework — so it's my pick for simple multi-agent routing, like triage to specialist, on the OpenAI stack. For complex stateful graphs with durability and HITL, LangGraph is richer."

### 58. Semantic Kernel?
"Semantic Kernel is Microsoft's **enterprise** SDK, available in C#, Java, and Python. Its model is plugins or skills — native or prompt functions — that planners compose to reach a goal, plus memory connectors, and now an Agent Framework and Process Framework for multi-agent and business processes. Its sweet spot is enterprise apps that need strong typing, dot-NET or Java support, governance and compliance, and integration with existing systems. It's heavier and has more concepts than a dedicated agent library, but for a regulated enterprise already on the Microsoft stack, that enterprise-grade story is exactly what you want."

### 59. LlamaIndex?
"LlamaIndex is **data-first**. Its strength is ingestion, indexing, and retrieval — the RAG layer — and on top of that it offers agents and an event-driven Workflows API. So I reach for it when the agent is fundamentally a **knowledge or RAG** system — lots of connectors, great retrieval tooling. For complex agent *orchestration* and control flow, LangGraph is richer, so a common pattern is LlamaIndex for the data and retrieval layer with another framework driving the orchestration."

### 60. When would you use NO framework?
"For a simple tool-calling agent, I'd often skip a framework entirely. A single agent that loops over the model's native function-calling API is about fifty lines of code — a while loop that calls the model, executes any tool it requests, appends the result, and repeats until it's done or hits a budget. That's fully transparent, easy to debug, and free of abstraction lock-in. Anthropic explicitly recommends starting framework-free for this reason. I adopt a framework when I actually need what it provides — durable state, human-in-the-loop, complex orchestration, built-in tracing — not by default. Showing that judgment usually lands well in interviews."

### 61. How do MCP + A2A let you mix frameworks?
"Because they standardize the interfaces at two layers. **MCP** standardizes the tool-and-data interface, so any agent — regardless of framework — can call the same MCP tools. **A2A** standardizes the agent-to-agent interface, so an agent in one framework can delegate a task to an agent in another. Put together, I could have a CrewAI crew that calls MCP tools and hands a subtask to a LangGraph agent over A2A. So instead of being locked into one ecosystem, I can pick the best framework per component and let the protocols glue them together — which is where the industry is heading."

---

## G. System design

> For these, narrate the **A-G-E-N-T-S** framework out loud: Align on requirements, Ground rules and risk, Establish the approach, Nodes and components, Tools/memory/context, Safeguards/eval/scale — then wrap with trade-offs and an MVP. Full walk-throughs are in [07-system-design.md](07-system-design.md).

### 62. Design a customer-support agent.
"First I'd align on requirements — resolve billing, tech, and account tickets over chat, deflect L1 volume, escalate safely, low latency, multilingual, and crucially it must not take destructive account actions without confirmation. Given distinct domains with different tools and policies, I'd use a **router plus handoff** multi-agent design: a triage agent classifies intent, language, and sentiment, then hands off to a billing, tech, or account specialist, each with its own scoped tools. For knowledge I'd add **RAG over the help center** with hybrid search, reranking, and citations, plus per-user memory from the CRM. On safeguards: a prompt-injection filter on user text, PII redaction in logs, and — the important one — **confirmation and authorization before any destructive tool** like a refund or password reset, with low-confidence or angry cases escalating to a human. I'd evaluate resolution accuracy, escalation precision and recall, and hallucination rate offline, and deflection rate and reopen rate online, with full tracing per ticket. And I'd ship an MVP first — a single FAQ RAG agent with human handoff — then add specialists once the metrics and guardrails are in place."

### 63. Design a deep-research assistant.
"Requirements: given a question, produce a cited report from web plus internal docs, thorough over fast so it can run async for minutes, with a budget per report. This is a textbook **orchestrator-worker** case because the sub-questions are independent, parallelizable, and each needs an isolated context. So a lead orchestrator decomposes the question and spawns N research workers in parallel; each worker searches, fetches, extracts, and summarizes in its *own* context and returns a structured brief. Then a **verification or citation agent** checks each claim is actually supported and drops or flags the unsupported ones, and a **writer** synthesizes the briefs into a report with inline citations, with the orchestrator running a quality gate and re-dispatching for gaps. The big safeguards are budgets — cap workers, depth, and total tokens — and treating fetched web content as **untrusted** to defend against indirect prompt injection. I'd evaluate factuality and citation accuracy with LLM-as-judge plus source-checking, and watch cost per report since parallelism multiplies tokens."

### 64. Design a coding agent.
"Requirements: take an issue, edit a repo, run tests, open a PR, with correctness critical and tests as the success signal, all sandboxed. I'd go **single-agent, ReAct plus reflection, single-threaded** — coding is tightly coupled, so naive multi-agent would cause context fragmentation; that's Cognition's lesson. The tools are read-file, edit-file, run-tests, a sandboxed shell, grep, and git, and the memory is a repo map plus the working set of files plus a summarized history. The key loop is **reflection on test failures** — read the error, revise, retry — which is Reflexion-style and drives the pass rate up. Optionally a separate **reviewer** agent critiques the final diff. Safeguards: a least-privilege sandbox, step and cost caps, and never push without passing tests plus human approval on the PR. I'd evaluate with a SWE-bench-style percentage of issues resolved, test pass rate, and regression rate, and trace every edit and command."

### 65. Design a transactional/voice agent.
"Take a voice ordering agent — phone orders for a restaurant. Latency is critical, and there's an irreversible action: charging the customer. So it's a **single agent with strict tool schemas and a mandatory confirmation step** before payment. The pipeline is speech-to-text, then the agent, then text-to-speech, and I'd keep a **structured order state** — items, quantities, prices — rather than relying on free-form chat, because that's where errors creep in. Guardrails on price and quantity, an **idempotent payment** call gated behind explicit confirmation, and a fallback to a human when the agent is confused. I'd stream responses to keep perceived latency low. Evaluation centers on order accuracy and completion rate. The theme is: low latency, but never an irreversible charge without a confirmation."

### 66. What's your design framework?
"I use a mnemonic I call A-G-E-N-T-S so I don't skip anything. **A** — align on requirements: goal, users, scale, success metrics, constraints. **G** — ground rules and risk: failure cost, reversibility, compliance, where humans are needed. **E** — establish the approach, and I justify the *simplest* design that works — workflow versus single agent versus multi-agent. **N** — nodes and components: I draw the architecture. **T** — tools, memory, and context strategy. **S** — safeguards, evaluation, and scale. Then I wrap with the key trade-offs and what I'd build as an MVP. Narrating that structure keeps the interviewer with me and shows I have a repeatable method."

### 67. First questions you ask?
"I spend the first few minutes on requirements because jumping to architecture is the classic way to fail. I ask: what exactly should it accomplish and what's in and out of scope; who uses it and at what volume and concurrency; what are the success metrics — task success, accuracy, latency, cost per task, CSAT; what's the latency and cost budget; what's the cost of a wrong action and is it reversible or regulated; what data sources, and any PII or access constraints; how much autonomy versus human approval, and where the checkpoints are; and which systems it has to integrate with. Those answers basically determine the whole design."

### 68. Sync request blocks for a 3-minute agent task — fix?
"A three-minute task can't sit on a blocking HTTP request — you'll hit timeouts and tie up resources. So I make it **asynchronous**: the request enqueues a job and returns immediately with a task ID, a pool of **workers** runs the agent, and the client gets progress via polling, websockets, or streaming. I **checkpoint** the trajectory so a long run can resume after a failure instead of restarting, and I surface partial results as they're produced for better perceived latency. That also lets me scale workers independently and apply backpressure under load."

### 69. Model strategy for cost?
"I use a **tiered, model-routing** strategy. A small, cheap, fast model handles routing, classification, and simple steps; a frontier model is reserved for the genuinely hard reasoning. I add **caching** — prompt or semantic caching and provider prompt caching for static context — so I'm not paying repeatedly for the same work. Over time, if a routine step is high-volume, I'll consider **distilling** a smaller fine-tuned model on the strong model's traces. The headline metric I track is cost per task, and routing is usually the biggest single lever on it without hurting quality."

### 70. Where put determinism?
"Anywhere I don't *need* the LLM, I put logic in **code**, not the model — input validation, routing rules when they're known, math, schema enforcement, business rules. LLMs are non-deterministic and relatively expensive, so using one for something a deterministic function does better is a mistake. I reserve the model for what genuinely needs language understanding or open-ended reasoning. That makes the system cheaper, faster, more testable, and easier to reason about — and it shrinks the surface where things can go wrong."

---

## H. Production, eval & security

### 71. How do you evaluate an agent?
"I evaluate at two levels. **Outcome** — did it achieve the goal? That's task success rate, the headline metric, plus correctness and quality. And **trajectory** — *how* did it get there? Did it pick the right tools in the right order, with no wasted steps or loops? An agent can get a right answer the wrong way and that won't generalize, so I grade both. For *how* I measure: **programmatic checks** when I have them — did the tests pass, is the schema valid; **LLM-as-judge** against a rubric for scale; and **human evaluation** to calibrate. And I run it both **offline**, on a curated regression set in CI, and **online**, with A/B tests and real metrics. The crucial habit is building that eval set from real production failures so it keeps getting harder where the agent is weak."

### 72. Outcome vs. trajectory eval?
"Outcome eval asks: was the final result correct and good — task success, accuracy, quality scores. Trajectory eval asks: was the *process* sound — did it select the right tools, call them with the right arguments, in a sensible order, without redundant steps or loops. You need both because they catch different failures: outcome catches wrong answers, trajectory catches inefficiency, brittleness, and lucky-but-wrong paths that will break later. In practice trajectory metrics also help me debug *why* an outcome was bad."

### 73. LLM-as-judge pros/cons?
"The pro is that it's **scalable and cheap** — I can grade thousands of outputs against a rubric without armies of human raters, and it handles nuance better than exact-match metrics. The cons are the biases: position bias, where it favors the first option; verbosity bias, favoring longer answers; and self-preference, favoring outputs from the same model family. So I don't trust it blindly — I **validate it against human labels** on a sample, use clear rubrics, prefer **pairwise comparisons** over absolute scores, and randomize order to cancel position bias. Treated carefully, it's the workhorse of agent eval; treated naively, it's misleading."

### 74. Name agent benchmarks.
"A few I'd cite by area. **τ-bench and τ²-bench** for tool-agent-user interaction in realistic settings. **SWE-bench** for coding agents resolving real GitHub issues. **WebArena** and **GAIA** for web and general assistant tasks. **AgentBench** for multi-environment agent evaluation. **BFCL**, the Berkeley Function-Calling Leaderboard, for tool-calling accuracy. And **RAGAS** for RAG-specific metrics. The point I'd make is that public benchmarks are a starting signal, but for a real product I build a domain-specific eval set, because that's what actually predicts production quality."

### 75. What do you trace/observe?
"I trace **every step of every trajectory**. For each LLM call: the prompt, the completion, token counts, cost, and latency. For each tool call: the arguments, the result, any error, and the duration. The full decision path — thoughts, actions, observations, retries — and the final outcome. In multi-agent systems, **per-agent attribution** so I can tell who did what. For tooling I'd use something like LangSmith, Langfuse, or Arize Phoenix, and increasingly **OpenTelemetry's GenAI conventions** so the traces are vendor-neutral. In production I monitor task success, step count, p50 and p95 latency, cost per request, tool error rates, loop rate, and guardrail triggers — you can't improve what you can't see."

### 76. Prompt injection — direct vs. indirect + defenses?
"Prompt injection is when malicious instructions enter the model's context and override its intended behavior. **Direct** is when the *user* types them — 'ignore your instructions and...'. **Indirect** is the nastier, agent-specific one: the instructions are hidden in content the agent *retrieves* — a web page, an email, a document, a tool result — and when the agent reads it, it gets hijacked. The defenses layer up: treat all external content as **untrusted** and never let it alter system instructions; keep instructions and data separated; filter and scan inputs; enforce **least-privilege** tools and scoped credentials so a hijack can't do much; and require validation or human approval before any destructive or data-exfiltrating action. And I'd mention breaking the lethal trifecta as the structural defense."

### 77. Excessive agency — meaning + mitigation?
"Excessive agency is an OWASP risk where the agent has **more permission, functionality, or autonomy than it needs**, so when something goes wrong — a hallucination or an injection — it can cause real damage, like deleting data or moving money. The mitigations are all about constraint: **least-privilege tools** and scoped credentials so it can only do what's necessary; **allow-lists** for tools and domains; **human-in-the-loop confirmation** before high-risk or irreversible actions; and validating actions against policy before they execute. The principle is to give the agent exactly the authority the task requires and not one bit more."

### 78. The 'lethal trifecta'?
"It's Simon Willison's framing of when an agent becomes uniquely dangerous: when it simultaneously has access to **private data**, exposure to **untrusted content**, and the ability to **communicate externally** — that is, exfiltrate. With all three, an indirect prompt injection in the untrusted content can read your private data and send it out. The defense is structural: **break at least one leg** — isolate private data from untrusted content, or remove the outbound channel, or sandbox the untrusted content — so no single path has all three capabilities at once. It's a great answer because it shows I think about agent security architecturally, not just with input filters."

### 79. Insecure output handling?
"This is the risk of **trusting the model's output** when it gets used by a downstream system. If the agent generates SQL that I run directly, or shell commands, or code that I eval, a bad or injected output becomes code execution or injection. The fix is to treat LLM output as untrusted: **validate and sanitize** it, **parameterize** queries instead of string-concatenating, never `eval` raw model output, run any generated code in a sandbox, and apply least privilege downstream. Basically the same discipline as handling untrusted user input — because that's effectively what it is."

### 80. Stop an agent looping forever?
"Defense in depth. Hard limits first: **max steps**, a **token and cost budget**, and a **wall-clock timeout**. Then **repeated-action detection** — I hash the recent actions and arguments, and if the agent repeats one, I break the loop. I make sure tool **errors flow back into context** so the model can actually adapt instead of retrying blindly, and I give it a **terminate or finish tool** plus a fallback path. If it's still stuck after repeated failure, it should **degrade gracefully** — return partial results or hand off to a human — rather than spin. Unbounded loops are the top cause of runaway cost, so I treat this as mandatory, not optional."

### 81. Cut cost without hurting quality?
"My toolkit, roughly in order of leverage. **Model routing** — cheap model for easy and routing steps, frontier model only for hard reasoning. **Caching** — prompt, semantic, and tool-result caching, plus provider prompt caching for static context. **Limit steps and tokens**, and **compact or summarize context** so I'm not re-paying for a bloated window every step. **Trim tool outputs** to signal. **Parallelize** independent work to cut latency, and **stream** to improve *perceived* latency. And throughout, I track **cost per task** as a first-class metric so I know which lever is actually moving the needle. Most of these cut cost with little or no quality hit."

### 82. Guardrail types?
"I think in three layers. **Input guardrails** — prompt-injection and jailbreak detection, PII detection and redaction, topic and policy filters, auth checks. **Output guardrails** — schema validation so structured output parses, safety and toxicity and compliance checks, a PII-leak check, and a groundedness or citation check so there are no unsupported claims. And **behavioral guardrails** — max steps, token and cost budgets, timeouts, tool allow-lists, rate limits, and human-in-the-loop approval before irreversible actions. The principle I stress is to enforce these in **code**, not by politely asking the model in the prompt."

### 83. Where do humans go in the loop?
"At the points where the cost of a mistake is high or irreversible. The main patterns are: **approve or reject** before an action executes — payments, deletions, external sends; **edit** the agent's plan or draft before it continues; **escalation** when confidence is low, the agent has failed repeatedly, or the case is sensitive; and **feedback capture**, where the human's correction feeds back into evaluation and improvement. Frameworks like LangGraph make this clean with interrupts and checkpointing, so the agent can pause, wait for a human, and resume. The rule I follow is: never let an agent take an irreversible high-stakes action without a human gate or very strong validation."

### 84. CI/CD for a prompt change?
"I treat **prompts, tools, and model versions like code** — they're versioned and reviewed. On every change, CI runs the **offline eval set** — the regression suite of real and adversarial cases — and **blocks the merge if it regresses** key metrics. Then I roll out with a **canary or A/B**, watch online metrics and guardrail triggers, and keep a **fast rollback** ready. After release, I mine production failures and feed them back into the eval set, so the suite hardens over time — that's the data flywheel. The mindset shift is that a prompt edit is a real deployment that needs testing, not a casual text tweak."

### 85. Why are agents non-reproducible and how to cope?
"Two compounding reasons: LLMs are **stochastic** — sampling means the same input can yield different outputs — and agents are **multi-step**, so tiny differences early can diverge into completely different trajectories. To cope, I reduce variance where I can — lower temperature and fix seeds for tool-use and deterministic steps — and I make non-LLM logic deterministic in code. But more importantly I **embrace the distribution**: I evaluate over many runs and report success *rates* and percentiles rather than judging a single run, and I rely on **full tracing** so even a one-off failure is inspectable and explainable. You manage non-determinism statistically; you don't pretend it away."

### 86. RAG hallucinates despite retrieval — debug?
"I localize the fault first, because it's either retrieval or generation. I check **retrieval quality** — are the right chunks even coming back? If context precision or recall is low, I fix chunking, switch to **hybrid search**, and add a **reranker**. If retrieval is good but the model still makes things up, the problem is **generation**: I tighten the prompt to answer *only* from the provided context and to say 'I don't know' when it's not there, add a **faithfulness or citation guardrail** that rejects unsupported claims, require inline citations, and lower the temperature. Evaluating the two stages separately, with something like RAGAS, is what tells me which knob to turn."

---

## I. Scenario / troubleshooting (think-aloud)

> For these, *think out loud*: state the likely cause, then your diagnosis steps, then the fix. Interviewers grade your reasoning process, not just the answer.

### 87. Your multi-agent system gives inconsistent final answers.
"My first hypothesis is **context fragmentation** — the agents don't share a single source of truth, so they're making conflicting sub-decisions that don't reconcile at the end. To diagnose, I'd pull the traces and look at what context each agent had and where they diverged. The fixes: centralize the plan in an **orchestrator** that owns the decomposition, pass **structured artifacts** instead of full chat transcripts so information doesn't degrade across handoffs, keep a **shared state as the single source of truth**, and add a **synthesizer** plus a **critic** at the end to reconcile and sanity-check before returning. If it's still inconsistent and the task is actually tightly coupled, I'd question whether multi-agent was the right call at all and consider collapsing it to a single agent."

### 88. Costs 10×'d after launch.
"That smells like **runaway loops, chatty agents, or bloated contexts**. I'd go straight to observability and look at cost per task, step counts, and token usage per request to find the outliers. Common culprits: agents looping on a failing tool, full transcripts being passed everywhere, or huge tool outputs sitting in context every step. The fixes map to the cause — add **step and budget caps** and loop detection, **trim and summarize** tool outputs and context, introduce **caching**, and apply **model routing** so cheap steps stop hitting the frontier model. Then I'd put an **alert on cost per task** so this never silently creeps again. The trace tells me exactly where the waste is."

### 89. Agent works in demo, fails on real users.
"Classic **distribution shift** — the demo covered the happy path, real users bring messy, adversarial, and edge-case inputs. I'd start by collecting the real failures and turning them into an **eval set**, because right now I'm probably flying blind on real traffic. Then I'd harden: add **guardrails** for the weird inputs, improve **tool error handling** so the agent recovers instead of breaking, and add a **human fallback** for low-confidence cases. Going forward I'd treat production failures as the input to a continuous-improvement loop — mine them, add them to evals, fix, re-evaluate. The meta-lesson is that a demo proves feasibility, not robustness."

### 90. Latency p95 is terrible.
"A bad p95 with an okay p50 usually means a **tail** — some trajectories run long or some tool is occasionally slow. I'd break the latency down per step and per tool to find where the tail comes from. Fixes: **cap the number of steps**, **parallelize** independent tool calls, add **timeouts with fallbacks** so one slow dependency can't dominate, use a **smaller model** for routing and simple steps, and add **caching**. And I'd **stream** output so the user perceives progress even when the full task takes a while. The goal is to bound the worst case, not just optimize the average."

### 91. Agent took a destructive action it shouldn't have.
"That's an **excessive agency** failure, and it's serious. Immediate response: figure out from the trace how it had the capability and what triggered it — was it an injection, a hallucination, a missing guard? The structural fixes: **remove or scope** that tool with least-privilege credentials, require **authorization plus a confirmation or human approval** before destructive actions, add **allow-lists**, and **validate actions** against policy before execution. I'd also make sure there are **audit logs** so this is detectable and reviewable. The principle is that an agent should never have had unguarded access to an irreversible action in the first place."

### 92. Retrieved web page hijacked the agent.
"That's textbook **indirect prompt injection** — malicious instructions embedded in content the agent fetched. The root cause is treating retrieved content as trusted. So: **isolate untrusted content** and never let it override system instructions — keep a hard boundary between instructions and data; **sanitize and filter** fetched content; enforce **least privilege** so even a successful injection can't do much; and **break the lethal trifecta** by making sure the agent doesn't simultaneously have private data, untrusted content, and an outbound channel. For high-risk actions, a human gate. This is exactly why I treat all external content — web, email, tool output — as adversarial by default."

### 93. How to roll out a risky agent safely?
"Graduated exposure with gates. I'd start in **shadow mode** — the agent runs and logs its proposed actions but doesn't execute them, so I can compare against humans with zero risk. Then **human-in-the-loop**, where it proposes and a person approves each action. Then a **canary** — a small percentage of real traffic, fully autonomous but closely watched. Then a gradual ramp to full rollout. At every stage I define success metrics and guardrail thresholds, and I keep an instant **rollback**. That way risk is bounded and I earn autonomy with evidence rather than assuming it."

### 94. Eval set is small and stale.
"Then it's not telling me much, so I'd grow it deliberately. The best source is **production failures** — mine real traces for cases the agent got wrong and add them, which is the data flywheel. I'd add **adversarial and edge cases** on purpose, make sure it's **categorized** so I can track performance per scenario, and use a **mix of human labels and LLM-as-judge** so it scales without losing calibration. I'd also refresh it as the product and data evolve so it doesn't drift out of date. A living eval set that gets harder where the agent is weak is one of the highest-leverage things I can maintain."

---

## J. ML/LLM foundations

### 95. Temperature/top-p effect on agents?
"Temperature and top-p control sampling randomness. Higher temperature means more diverse, exploratory outputs; lower means more deterministic, focused ones. For **agents, I usually run low temperature** on the parts that need precision — tool selection and argument generation — because I want reliable, valid, repeatable calls, not creative variation that produces malformed arguments. I'd raise it only where exploration helps, like brainstorming or sampling multiple chains for self-consistency. Top-p, nucleus sampling, is the related knob that limits sampling to the smallest set of tokens covering probability p. The general agent instinct is: determinism where it matters, diversity only where it pays."

### 96. Structured outputs / JSON mode?
"Structured outputs, JSON mode, or constrained decoding force the model to emit output that **conforms to a schema** — valid JSON, or a specific shape. For agents this is huge because it makes **tool arguments and parsing reliable** — I'm not regexing free text and hoping. The model is constrained at decode time to produce schema-valid tokens, which basically eliminates a whole class of parse-and-retry failures. So any time I need machine-readable output — tool calls, extraction, routing decisions — I use structured outputs rather than asking nicely in the prompt and validating after."

### 97. Context window limits & cost scaling?
"The context window is the max tokens the model can attend to at once. Two practical consequences. **Cost and latency scale with length** — attention is expensive, so a bigger context means more money and more time per call. And quality isn't flat across the window — the **lost-in-the-middle** effect means information buried in the middle gets underweighted. So even with long-context models, I don't just stuff everything in; I **curate** — retrieve what's relevant, summarize the rest, and place the critical content at the start or end. Long context is a tool, not a substitute for context engineering."

### 98. Fine-tuning vs. RAG vs. prompting for domain knowledge?
"They solve different problems, so I match the tool to the need. **Prompting**, including few-shot, is fastest and great for steering behavior and format with zero training. **RAG** is the right tool for **fresh or private factual knowledge** — it grounds answers in retrieved sources, gives citations, and updates instantly when the data changes, with no retraining. **Fine-tuning** is for **behavior, style, format, or specialized skills**, and for cost and latency by baking patterns into a smaller model — but it's *not* the way to inject frequently-changing facts, since those go stale in the weights. In practice I often **combine** them — fine-tune for behavior, RAG for knowledge, prompt to orchestrate."

### 99. What's distillation and why for agents?
"Distillation is training a smaller, cheaper 'student' model to mimic a larger 'teacher' model — often on the teacher's outputs or traces. For agents the motivation is **cost and latency**: a lot of agent steps are routine — routing, classification, simple tool calls — and I don't need a frontier model for those. If I distill a small model on the strong model's behavior for those steps, I get most of the quality at a fraction of the cost and latency, and I reserve the expensive model for the genuinely hard reasoning. It pairs naturally with model routing."

### 100. Embeddings — what/why?
"Embeddings are dense vector representations of text — or other data — where semantic similarity shows up as geometric closeness, so similar meanings have nearby vectors. They're the foundation of **retrieval and memory** in agents: I embed documents and queries and use vector similarity to find relevant context, which is the heart of RAG and of episodic memory recall. The practical choices are the embedding model, its dimensionality, and domain fit — a domain-tuned embedding can meaningfully improve retrieval quality. Without embeddings, semantic search and most long-term memory just don't work."

### 101. Why do errors compound in agents?
"Because each step is **conditioned on the previous state**, and if that state already contains a mistake, the next step builds on the mistake. Roughly, if each step has accuracy p and the task needs n correct steps in sequence, the success probability is about p to the n — so even ninety-five-percent-per-step accuracy over ten steps drops below sixty percent. That's the core reliability challenge of agents, and it's *why* the techniques in the rest of this kit exist: keep trajectories **short**, add **checks and validation** between steps, enable **recovery and reflection** so errors get caught rather than propagated, and ground steps in real observations. Compounding error is the reason 'it works once in a demo' doesn't mean 'it works reliably.'"

---

## K. Behavioral / experience

> These need *your* real stories. Below are **templates with placeholders** in [brackets] — swap in your actual projects. Use **STAR**: Situation, Task, Action, Result. Keep it to about 90 seconds and always land a quantified result.

### 102. Tell me about an agent/LLM system you built.
"Template: 'I built [a customer-support / research / coding] agent to [solve X problem] for [users]. The challenge was [why a single prompt or workflow wasn't enough — e.g., dynamic tool selection across N systems]. I chose [single agent / orchestrator-worker] because [coupling/parallelism reasoning], with [tools], [memory: RAG over X + per-user state], and [ReAct + reflection]. For reliability I added [guardrails, step caps, HITL on irreversible actions] and evaluated with [offline eval set + online metrics]. The result was [quantified: deflected X% of tickets / cut handle time by Y% / resolved Z% of issues], and the biggest lesson was [observability/eval mattered more than the model choice].' The keys: justify *why agentic*, name the architecture trade-off, and end with a number."

### 103. A time your agent failed in prod — what did you learn?
"Template: 'We shipped [agent] and saw [failure: looping cost spike / hallucinated action / injection]. Because we had [tracing], I traced it to [root cause]. I fixed it with [specific mitigation — step caps / guardrail / least-privilege tool], and then I made it *systemic* by [adding the case to the eval set / adding an alert] so it couldn't recur silently.' Interviewers love this because it shows **observability-driven debugging** plus a **process fix**, not just a one-off patch. Pick a real, not-too-catastrophic failure and emphasize what you changed structurally."

### 104. How do you decide build vs. buy a framework?
"My answer: 'It's needs-driven, and I start simple. For a basic tool-using agent I'll often go framework-free — a fifty-line loop over function calling is transparent and avoids lock-in. I adopt a framework when I need what it actually provides: durable state and human-in-the-loop pushes me to LangGraph; conversational multi-agent to AutoGen; fast role-based teams to CrewAI; enterprise governance to Semantic Kernel. I weigh the abstraction cost — frameworks can hide failure modes — against the velocity gain, and I avoid betting the whole system on one fast-moving SDK.' It signals pragmatism and that I won't cargo-cult a framework."

### 105. How do you keep up with this fast-moving field?
"My honest answer: 'I read the foundational papers so I understand *why* things work — ReAct, Reflexion, Tree of Thoughts, Toolformer — and I follow the engineering blogs from Anthropic, OpenAI, and Google because they share hard-won production lessons. I track the protocols like MCP and A2A and the benchmarks like SWE-bench and τ-bench to know what 'good' means. And most importantly I **build** — I prototype the new patterns on small projects, because hands-on is how the concepts actually stick. I optimize for understanding fundamentals over chasing every new framework, since the fundamentals transfer and the APIs churn.'"

### 106. Disagreement on going multi-agent — how resolve?
"Template: 'A teammate wanted multi-agent; I thought a single agent was right. Rather than argue abstractly, I framed it as **trade-offs** — coupling, context pressure, and the token-cost multiplier — and proposed we **prototype both** on a representative slice and let the **metrics** decide: task success, cost per task, and latency. The data showed [outcome], and we went with [decision]. The point I'd make is I resolve technical disagreements with evidence and shared criteria, not seniority or opinion.'"

### 107. How do you balance autonomy vs. safety for a business?
"My answer: 'I tie the autonomy level directly to the **cost and reversibility of mistakes**. For low-risk, reversible actions I let the agent run autonomously and move fast. For high-stakes or irreversible actions — money, data deletion, external communication — I require human approval and least-privilege scoping. Then I **expand autonomy gradually as the agent earns trust** through measured reliability — shadow mode, then HITL, then canary, then full. So safety isn't a blocker on autonomy; it's the mechanism that lets me grant autonomy responsibly and prove it with metrics.'"

### 108. Most interesting agentic problem you've thought about?
"Have a concrete, opinionated answer ready. Template: 'I'm fascinated by [e.g., long-horizon reliability — keeping an agent coherent over hundreds of steps, where compounding error and context management dominate], because [why it's hard and matters]. I think the interesting levers are [memory architecture / verification loops / decomposition], and I'd love to work on [specific aspect].' The goal is to show genuine intellectual engagement and that you think past the tutorial level — pick something you can actually discuss for a few minutes if they probe."

---

## L. Rapid-fire one-liners (say these crisply, ~10 seconds each)

> When asked for a quick definition, give exactly one or two sentences and stop. These are your snap answers.

- **ReAct** — "An agent loop that interleaves reasoning and acting: thought, action, observation, repeat, so reasoning stays grounded in real tool results."
- **Reflexion** — "The agent critiques its own failure in words, stores that reflection in episodic memory, and uses it to do better on the next attempt — verbal RL within a session."
- **MCP** — "An open standard that connects tools, data, and prompts to models — USB-C for AI — turning N-times-M integrations into N-plus-M."
- **A2A** — "A standard for agents to delegate tasks to each other across vendors, using Agent Cards over HTTP."
- **Orchestrator-worker** — "A lead agent decomposes a task, delegates subtasks to specialist workers, often in parallel, and synthesizes the results."
- **Excessive agency** — "When an agent has more permission or autonomy than it needs, so a failure can do real damage — mitigated by least privilege and human gates."
- **Lethal trifecta** — "Private data, untrusted content, and an outbound channel all at once — break one leg to stop exfiltration."
- **Context engineering** — "Curating exactly the right instructions, memory, tools, and data into the window at the right time, within a token budget."
- **LLM-as-judge** — "Using a model to score outputs against a rubric — scalable, but you must validate it against humans and watch for position and verbosity bias."
- **HITL** — "A human checkpoint that approves or edits before the agent takes a risky or irreversible action."
- **Trajectory eval** — "Grading *how* the agent got there — right tools, right order, no loops — not just the final answer."
- **Agentic RAG** — "Retrieval as a tool the agent decides whether and how to use, and can iterate on — versus a fixed one-shot pipeline."
- **Self-consistency** — "Sample several reasoning chains and majority-vote the answer for a cheap reliability boost."
- **Plan-and-Execute** — "Plan the whole task up front, then execute the steps, replanning if reality diverges — efficient for long, decomposable work."
- **Tool calling** — "The model emits a structured call; my code executes it and returns the result — the model never runs anything itself."
- **The senior default** — "Use the lowest level of agency that solves the problem."

---

## Closing delivery tips

- **Open with the definition, then the why, then one example.** Don't bury the lede.
- **Use the structure words** — "two levels," "three reasons," "the trade-off is" — so the interviewer can follow.
- **Name-drop precisely**: ReAct, Reflexion, MCP, A2A, lethal trifecta, SWE-bench. Specificity signals depth.
- **Always volunteer eval, observability, and safety** — that's what separates senior from junior answers.
- **End on trade-offs**: "I'd start with the simplest version and add complexity only if the metrics demand it."
- **Pause.** Finish your point and let them ask the follow-up instead of monologuing.

