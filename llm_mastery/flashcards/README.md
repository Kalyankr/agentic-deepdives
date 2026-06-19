# Flashcards — Anki decks

Spaced-repetition material distilled from the 8 stage interview banks plus the [System Design](../system-design/README.md) module (the full Q&A lives in each stage's `interview-questions.md` + `answers.md`).

| File | Cards | Format | Use it for |
|------|-------|--------|------------|
| [llm-mastery-flashcards.csv](llm-mastery-flashcards.csv) | ~85 | Basic Q→A | Fast high-yield review — the must-know facts/formulas. Start here. |
| [llm-mastery-flashcards-full.csv](llm-mastery-flashcards-full.csv) | 399 | Basic Q→A | One card per interview question across all 8 stages plus a System Design track — full coverage. |
| [llm-mastery-cloze.csv](llm-mastery-cloze.csv) | 62 | **Cloze** | Drill exact formulas/derivations by filling in the blanks. |
| [printable-study-sheet.md](printable-study-sheet.md) | 399 | Markdown | Offline / print: all cards as a Q→A sheet, grouped by stage + System Design. |

Each card is a single fact/formula/definition/approach so it's easy to recall and grade.

## Import into Anki
1. Anki → **File → Import** → pick the `.csv`.
2. The file's header lines (`#separator:Comma`, `#columns`, `#tags column:3`) auto-configure the mapping: **Field 1 → Front, Field 2 → Back, Field 3 → Tags**.
3. Choose/create a deck (e.g. `LLM Mastery`), set type **Basic**, and import.

> Import the curated deck and the full deck into the **same** `LLM Mastery` deck — the ~85 curated cards are a subset, so duplicates are merged on the Front field rather than doubled.

### Cloze deck
The cloze file declares `#notetype:Cloze`, so on import select the **Cloze** note type (columns map **Text, Back Extra, Tags**). Each `{{c1::…}}` blank becomes its own card, so a formula with three blanks drills as three cards.

### Printable sheet
[printable-study-sheet.md](printable-study-sheet.md) is plain Markdown — open it in VS Code preview and **Print → Save as PDF**, or run it through pandoc, for an offline study sheet. It's regenerated from the full CSV, so the CSV stays the single source of truth.

> Plain text is used (`#html:false`) so code and math symbols show literally. For nicer math, install an Anki LaTeX/MathJax add-on and the `$...$` notation in the source answers will render.

## Tag scheme (filter with the browser, e.g. `tag:stage5 tag:math`)
- `stage1`…`stage8` — which stage.
- `systemdesign` — the cross-cutting System Design track; sub-areas `chatgpt` · `rag` · `inference` (e.g. `tag:systemdesign tag:rag`).
- `fundamentals` · `core` · `senior` — difficulty tier (the full deck adds `senior` for staff-level questions).
- `math` · `stats` · `coding` · `debug` — angle.
- `moe` · `longcontext` · `reasoning` · `multimodal` · `interp` · `synthetic` — Stage 7 specialization tracks.

## Study tips
- Do **stage-by-stage** first (`tag:stage1`), then mix all once recall is solid.
- Drill one track at a time in Stage 7, e.g. `tag:stage7 tag:reasoning`.
- When a card is easy, open the matching `answers.md` entry and push one level deeper (the "why behind the why").
- Re-derive every `math` card on paper rather than recognizing it.

---
[← Back to index](../README.md)
