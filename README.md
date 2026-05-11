# Multi-Agent Coding Assistant

(Scroll down for Screenshots)<br>
An advanced multi-agent AI coding assistant built with LangGraph, hybrid RAG, long-term memory, and self-correcting generation pipelines.

This system ingests an entire project codebase, understands its structure semantically, retrieves relevant context intelligently, and orchestrates specialized AI agents to generate, explain, debug, review, and improve code with built-in verification and learning loops.

It combines concepts inspired by modern agentic systems such as Cursor, Devin, Copilot Workspace, Self-RAG, Corrective RAG, and blackboard-style distributed agent communication.

---

## Core Idea

Most AI coding tools fail because they:

* do not understand the real project structure,
* lose context across interactions,
* hallucinate APIs or functions,
* cannot validate their own output,
* and do not learn from previous mistakes.

This project solves those problems using:

* Multi-agent orchestration with LangGraph
* Hybrid retrieval (semantic + keyword)
* Self-evaluating RAG pipelines
* Corrective retrieval loops
* Long-term experience memory
* Verification and review layers
* Shared message-bus communication architecture
* ReAct-style tool-using coding agent

The result is a project-aware coding assistant that behaves less like a chatbot and more like a coordinated software engineering system.

---

# System Architecture

## High-Level Pipeline

```text
User Query
    ↓
Intent Classification
    ↓
Supervisor Routing
    ↓
Planner Agent
    ↓
Retriever Agent
    ├── Query Rewriting
    ├── Hybrid RAG
    ├── Self-RAG Evaluation
    └── Corrective RAG (if needed)
    ↓
Coder Agent (ReAct Loop)
    ↓
Syntax Verifier
    ↓
Reviewer Agent
    ├── APPROVED → Final Response
    └── REJECTED → Supervisor Retry Guidance → Coder Retry
    ↓
Learning Node
    ↓
Long-Term Memory Update
```

---

# Key Features

## Multi-Agent Orchestration

The system uses specialized agents instead of a single monolithic prompt.

Each agent has one focused responsibility:

| Agent             | Responsibility                                      |
| ----------------- | --------------------------------------------------- |
| Intent Classifier | Understands user intent (write/fix/explain/general) |
| Supervisor        | Central routing and retry coordination              |
| Planner           | Breaks tasks into structured execution steps        |
| Retriever         | Performs retrieval and memory collection            |
| Coder             | Generates or edits code using ReAct-style reasoning |
| Verifier          | Performs AST-based syntax validation                |
| Reviewer          | Critiques quality, correctness, and maintainability |
| Learning Node     | Stores lessons into long-term memory                |

This modular architecture improves reliability, observability, debugging, and scalability.

---

# Shared Message Bus Architecture

One of the most important design decisions in this system is the blackboard-style communication architecture.

Agents do not directly call each other.

Instead:

* Every agent reads shared state.
* Every agent appends summarized outputs to a shared message bus.
* LangGraph decides which node executes next.

This creates:

* Loose coupling
* Better traceability
* Easier debugging
* Scalable orchestration
* Safer extensibility

## AgentState

The entire pipeline operates on a shared `AgentState` object.

It contains:

```python
{
    "query": ...,
    "intent": ...,
    "plan": ...,
    "context": ...,
    "answer": ...,
    "review": ...,
    "messages": [...],
    "tool_calls": [...],
    "retrieval_confidence": ...,
    "supervisor_notes": ...
}
```

## AgentMessage Structure

Each agent appends structured summaries:

```python
{
    "agent": "retriever",
    "type": "context",
    "content": "Retrieved 6 relevant chunks",
    "metadata": {
        "confidence": 0.46
    }
}
```

This allows every downstream agent to understand:

* what already happened,
* what decisions were made,
* what quality signals exist,
* and what should happen next.

---

# Hybrid RAG System

The retrieval engine is one of the strongest components of the project.

Instead of relying only on embeddings, the system combines multiple retrieval techniques.

## Retrieval Pipeline

```text
User Query
    ↓
Query Rewriter
    ↓
BM25 Keyword Retrieval
    ↓
Chroma Semantic Retrieval
    ↓
Merge + Deduplicate
    ↓
Cosine Similarity Reranking
    ↓
Top Context Chunks
```

---

## Why Hybrid RAG?

Semantic search alone often misses:

* exact function names,
* variable names,
* APIs,
* class names,
* file references.

Keyword search alone often misses:

* conceptual similarity,
* implementation patterns,
* semantic intent.

Combining both produces significantly stronger retrieval quality.

---

# Self-RAG

After retrieval, the system evaluates its own retrieval quality.

This is implemented using cosine similarity scoring between:

* the user query embedding,
* and retrieved chunk embeddings.

## Purpose

The system should not blindly trust retrieval.

If the retrieved context is weak, irrelevant, or noisy, generation quality collapses.

## Retrieval Confidence

The system computes:

```text
retrieval_confidence = average cosine similarity
```

If confidence is high:

```text
retrieval_ok = True
```

If confidence is poor:

```text
retrieval_ok = False
```

This quality signal is later used by Corrective RAG.

---

# Corrective RAG

When Self-RAG detects weak retrieval, the system automatically retries retrieval using intent-aware reformulation.

## Example

Original query:

```text
"fix csv issue"
```

Corrective query:

```text
"bug error exception pandas csv parser fix handler"
```

Different intent types receive different corrective expansions:

| Intent  | Corrective Expansion                        |
| ------- | ------------------------------------------- |
| Write   | implementation / class / function / example |
| Fix     | error / bug / exception / handler           |
| Explain | logic / flow / structure                    |

The system then reruns Hybrid RAG using the improved query.

This reduces hallucinations and improves grounding.

---

# Query Rewriter

Natural language is often too vague for retrieval.

Example:

```text
"load csv"
```

The actual code may contain:

* `pd.read_csv`
* `DataFrame`
* `csv_loader`
* `preprocess_dataset`

The Query Rewriter expands the query using:

1. Static technical mappings
2. Project-specific co-occurrence learning
3. Global learned expansion patterns
4. Intent-aware contextual enrichment

This significantly improves retrieval recall.

---

# ReAct Coding Agent

The Coder Agent is not a one-shot generator.

It uses a ReAct-style loop:

```text
Reason → Act → Observe → Repeat
```

The coder can:

* request additional files,
* inspect full source files,
* ask for more context,
* then continue generation.

## Example Flow

```text
Coder:
"I need to inspect auth.py"

Tool:
Returns file contents

Coder:
Continues implementation using new context
```

This behavior mimics modern agentic IDE systems.

---

# Verification Layer

Before review, generated code passes through a syntax verification stage.

The verifier uses Python's built-in AST parser:

```python
ast.parse(code)
```

This provides:

* zero-cost validation,
* instant syntax feedback,
* early rejection of malformed code.

If syntax fails:

* the graph routes back to the Coder,
* only the syntax error is provided,
* and the coder performs focused correction.

---

# Reviewer Agent

The Reviewer Agent acts as a quality gate.

It evaluates:

* correctness,
* maintainability,
* readability,
* robustness,
* style,
* edge cases,
* architecture consistency.

The reviewer can:

```text
APPROVE
```

or:

```text
REJECT
```

Rejected outputs trigger the Supervisor Retry Node.

---

# Supervisor Logic

The Supervisor is the orchestration brain of the system.

## Responsibilities

### 1. Routing Decisions

The supervisor decides whether to:

* proceed normally,
* skip retrieval,
* or request additional context.

Example:

```text
General Python question?
→ Skip retrieval

Project-specific implementation?
→ Full RAG pipeline
```

---

### 2. Retry Coordination

When review fails, the supervisor converts review feedback into actionable repair guidance.

Example:

```text
Reviewer:
"Missing type hints and docstring"

Supervisor:
"Add filepath: str typing, return type, and function docstring"
```

This makes retries targeted instead of random.

---

# Memory System

The system contains multiple memory layers operating at different time scales.

---

## 1. Session Memory

Stored as JSON.

Purpose:

* maintain conversational continuity,
* remember recent tasks,
* preserve short-term workflow state.

Examples:

```text
Previously generated preprocessing pipeline
Explained family_type() earlier
```

Used mainly by the Coder Agent.

---

## 2. Experience Memory

Stored inside Chroma vector collections.

Contains:

* successful implementation patterns,
* reviewer-approved code,
* past mistakes,
* correction histories.

## Retrieval

The Retriever Agent fetches relevant experiences during generation.

The Coder then receives:

```text
Past mistakes to avoid
```

and:

```text
Previously successful patterns
```

This creates a self-improving feedback loop.

---

## 3. Project Code Memory

The uploaded project itself becomes a searchable semantic memory.

Python files are chunked intelligently using AST parsing:

* functions,
* classes,
* methods,
* logical blocks.

These chunks are embedded and stored in Chroma.

Retrieval happens during nearly every project-aware query.

---

# Knowledge Base

The system also includes an auxiliary knowledge base.

Stored as markdown documentation files.

Examples:

* NumPy
* Pandas
* Scikit-learn
* Python best practices
* common architecture patterns

This allows the system to combine:

```text
Project Context + Domain Knowledge
```

instead of relying only on uploaded code.

---

# LangGraph Workflow

The entire orchestration layer is implemented using LangGraph.

## Why LangGraph?

LangGraph enables:

* deterministic multi-agent pipelines,
* conditional routing,
* retry edges,
* persistent state passing,
* graph-based execution,
* observability.

## Pipeline Characteristics

* Stateful
* Deterministic
* Retry-aware
* Inspectable
* Extensible

---

# LangSmith Observability

Every important operation is traced using LangSmith.

This provides:

* full execution visibility,
* debugging support,
* agent reasoning inspection,
* retrieval analysis,
* token tracking,
* latency monitoring.

The complete lifecycle of a query can be inspected step-by-step.

---

# Technical Stack

## Backend

* Python
* FastAPI
* LangGraph
* LangChain
* ChromaDB
* BM25 Retrieval
* AST Parsing
* Sentence Transformers

## AI / Retrieval

* Hybrid RAG
* Self-RAG
* Corrective RAG
* ReAct Agent Architecture
* Embedding-based Retrieval
* Semantic Reranking

## Infrastructure

* Docker
* Docker Compose
* LangSmith Tracing
* Environment-based configuration

---

# Example Query Flow

## User Query

```text
"Write a function to preprocess CSV files"
```

## Internal Flow

1. Intent Classifier detects WRITE intent.
2. Supervisor routes to full retrieval pipeline.
3. Query Rewriter expands:

```text
csv → read_csv, DataFrame, preprocessing
```

4. Hybrid RAG retrieves:

* project preprocessing functions,
* related utilities,
* pandas examples.

5. Self-RAG scores retrieval confidence.
6. Corrective RAG triggers if confidence is weak.
7. Coder generates implementation.
8. Verifier checks syntax using AST.
9. Reviewer critiques quality.
10. Supervisor coordinates retry if needed.
11. Learning node stores lessons into memory.

---

# Design Philosophy

This project was intentionally designed around production-grade AI engineering principles:

* modular orchestration,
* explicit state management,
* grounded generation,
* retrieval quality control,
* observability,
* self-correction,
* memory persistence,
* and extensibility.

The goal was not simply to generate code, but to build a system capable of:

```text
reasoning over real software projects
```

while maintaining architectural transparency and operational reliability.

---

# Screenshots

## App
<img width="1919" height="916" alt="Screenshot 2026-05-11 174428" src="https://github.com/user-attachments/assets/22a209d7-7228-44d3-8f40-a6c80043d4b0" />
<img width="1919" height="909" alt="Screenshot 2026-05-11 174444" src="https://github.com/user-attachments/assets/f1fea112-2d52-458b-9aba-bac75d0531b5" />
<img width="1919" height="912" alt="Screenshot 2026-05-11 174457" src="https://github.com/user-attachments/assets/25fb82ad-d825-4978-9491-4c97bd6bae0d" />
<img width="1919" height="907" alt="Screenshot 2026-05-11 174515" src="https://github.com/user-attachments/assets/bd4ad90d-cb3c-4430-a350-cab4ad8b320d" />
<img width="1919" height="912" alt="Screenshot 2026-05-11 174530" src="https://github.com/user-attachments/assets/a5e23076-9358-445d-9a01-a78187ee454f" />
<img width="1668" height="951" alt="Screenshot 2026-05-11 174726" src="https://github.com/user-attachments/assets/a260ede7-8931-4e87-9353-47f94ad4ede7" />

## LangSmith Traces
<img width="1908" height="895" alt="Screenshot 2026-05-11 175119" src="https://github.com/user-attachments/assets/6a912958-8cdb-47e9-a076-e845cbda4b6a" />




---

# Future Improvements

* Parallel specialized retrieval agents
* Tool-augmented debugging agents
* Automatic test generation
* Multi-file refactoring support
* Persistent vectorized conversation memory
* Multi-language ingestion (Rust, Go, Java, C++)
* Autonomous repository analysis
* Git-aware commit planning
* Structured software architecture extraction
* GraphRAG integration

---

# What This Project Demonstrates

This project demonstrates understanding of:

* agentic AI system design,
* retrieval engineering,
* distributed orchestration patterns,
* memory systems,
* stateful workflows,
* production AI architecture,
* semantic search,
* vector databases,
* code intelligence pipelines,
* observability and tracing,
* and self-correcting generation systems.

---

# Repository Structure

```text
multi_agent_coder/
│
├── app/
│   ├── agents/
│   ├── services/
│   ├── routes/
│   └──main.py
│ 
│
├── knowledge/
├── docker-compose.yml
├── docker/
├── requirements.txt
└── README.md
```

---

# Closing Notes

This project explores how modern AI systems can move beyond simple prompting into coordinated, stateful, retrieval-grounded software engineering workflows.

It combines research-inspired ideas with practical engineering implementations to create a robust multi-agent code intelligence platform capable of understanding, generating, reviewing, correcting, and learning from real-world codebases.
