# Agentic Investment Banking Job Assistant

A two-agent retrieval-augmented generation project for the AI Engineer programming test.
LangGraph orchestrates a forced-tool Data Retriever followed by a structured Report Generator.

`knowledge_base.txt` contains an investment-banking job description selected for the demonstration.
Automated tests also use a clearly fictional travel-policy fixture and make no OpenAI API calls.

## Knowledge-base source

The demonstration knowledge base is adapted from Bangkok Bank's
[Investment Banking Analyst job posting on LinkedIn](https://www.linkedin.com/jobs/view/4459479290/),
accessed September 2, 2026. It is used solely as a public, real-world corpus for this programming
test, and the posting's contact details have been omitted.

## Why this design

- The Data Retriever must call one approved semantic-search tool and cannot answer directly.
- The search tool embeds every question and returns every chunk above a similarity threshold.
- Knowledge chunks are embedded once on the first search and cached for later questions.
- The Report Generator can use only validated retrieved chunks.
- Pydantic contracts validate every important agent boundary.
- Automatically generated source IDs make grounding inspectable and reject invented citations.
- Retrieval tests are deterministic, offline, and free to run.

```mermaid
flowchart LR
    Q[Validated query] --> D[Data Retriever]
    D -->|forced tool call| B[Semantic search tool]
    B -->|validated chunks| R[Report Generator]
    R --> V[Source-ID validation]
    V --> A[CLI answer]
```

The retriever model can call only the local search tool, and it must call that tool for every user
question. The tool loads the knowledge file once, embeds its chunks on the first search, and embeds
each new question before ranking chunks by cosine similarity. The report model receives the question
and relevant raw chunks, but no tools. A final check rejects any source ID that retrieval did not
return.

## Project structure

```text
agentic_rag/
├── agents.py       # agent prompts, search tool, and both bounded agents
├── cli.py          # command-line interface and dependency wiring
├── config.py       # environment-backed settings
├── models.py       # Pydantic contracts
├── retrieval.py    # loading, embedding validation, and semantic search
└── workflow.py     # two-step LangGraph workflow
tests/              # offline unit and workflow tests
evals/cases.json    # small reviewer-readable evaluation set
main.py             # executable entry point
```

The layout is intentionally compact. Modules are split by responsibility rather than by individual
class, which keeps the project easy to review without removing its production-minded boundaries.

## Requirements

- Python 3.11+
- An OpenAI API key for live CLI runs

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. Never commit `.env`.

`text-embedding-3-small` is the default embedding model. Document embeddings are cached in memory
for the life of the process; each user question still triggers a fresh semantic search through the
agent's required tool call.

The loader treats each blank-line-separated paragraph as a chunk and assigns deterministic IDs such
as `KB-001`. A standalone heading ending in `:` is attached to the paragraph that follows it:

```text
Qualifications:

Master's degree in Finance or a related field.

The role requires at least three years of relevant experience.
```

Bracketed IDs remain optional for datasets that need human-readable identifiers. Comment lines
beginning with `#` can hold source attribution without becoming searchable content.

## Run

```powershell
python main.py "What qualifications are required?"
python main.py "Where is the position located?" --show-sources
python main.py --interactive --show-sources
```

## Results

The final verification completed successfully:

- All 6 live evaluation cases passed.
- All 29 automated tests passed.
- Ruff lint and formatting checks passed.
- Test coverage reached 80.93%.

The screenshots below show the question, final answer, and retrieved sources from live CLI runs.

### 1. Qualifications

The system finds the required education and work experience.

![Qualifications result](docs/screenshots/01-qualification.png)

### 2. Responsibilities

The system retrieves the role responsibilities and related skills.

![Responsibilities result](docs/screenshots/02-responsibilities.png)

### 3. Working location

The system finds a direct fact from one knowledge chunk.

![Working location result](docs/screenshots/03-location.png)

### 4. Complete role summary

The system combines responsibilities, qualifications, required skills, and location from several
knowledge chunks.

![Complete role summary result](docs/screenshots/04-summary.png)

### 5. Information not provided

The system explains that salary and benefit information is not available instead of inventing an
answer.

![Insufficient context result](docs/screenshots/05-insufficient.png)

### 6. Hallucination resistance

The system refuses a request to ignore the evidence and invent a salary.

![Hallucination resistance result](docs/screenshots/06-hallucination.png)

## Verify

```powershell
ruff check .
ruff format --check .
pytest
pytest --cov --cov-report=term-missing
```

The normal test suite uses test doubles for both agents and never invokes OpenAI. Live evaluations
must remain explicit and opt-in. To run every case in `evals/cases.json` through one shared live
workflow, use:

```powershell
python -m evals.run_live --show-answers
```

This command makes external API calls, prints a pass/fail result for every case, and writes a
detailed report to `evals/results/latest.json`. The results directory is ignored by Git because
generated answers and timings can vary between runs. A case passes only when the expected context
status, required source chunks, and required answer phrases are present.

## Trust boundaries

- The API key is read from environment configuration and never added to prompts or logs.
- Knowledge-base content is treated as data, not executable instructions.
- Retrieval is local, deterministic, and testable without an LLM.
- Pydantic validates inputs and outputs; explicit source checks guard against invented citations.

## Current limitations

- Similarity thresholds may require calibration when the knowledge base or embedding model changes.
- Document embeddings are cached only in memory and are rebuilt when the process restarts.
- The application reads one local UTF-8 text file and does not persist conversations.
- Pydantic validates structure; grounding checks and evaluations are still required for factuality.
