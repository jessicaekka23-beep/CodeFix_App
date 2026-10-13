# CodePilot Local — Presentation Deck Outline (10 Slides)

**Academic Title:** A Privacy-Preserving Local AI Agent for Code Analysis, Debugging, and Automated Fix Generation  
**Presenter:** Jessica Maria Ekka  
**Context:** GenAI Coursework Project Presentation  

---

## Slide 1: Title and Introduction

- **Title:** CodePilot Local
- **Subtitle:** A Privacy-Preserving Local AI Agent for Code Analysis, Debugging, and Automated Fix Generation
- **Presenter:** Jessica Maria Ekka
- **Core Value Proposition:**
  - Privacy-first coding assistant with local inference by default and optional Groq Cloud acceleration.
  - Combines local Retrieval-Augmented Generation (RAG) with autonomous multi-step agentic workflows.
  - Preserves corporate IP and developer privacy while running on standard 16 GB RAM consumer hardware.

---

## Slide 2: Problem Statement & Motivation

- **The Cloud Dilemma:**
  - Transmitting proprietary codebases and stack traces to cloud endpoints breaches NDAs, GDPR, and enterprise security policies.
- **The Context Gap:**
  - Standard chatbots lack workspace context, generating hallucinations and ungrounded code suggestions.
- **The Unchecked Execution Risk:**
  - Applying LLM-generated code directly to workspaces without isolated testing risks breaking working environments.
- **Hardware Realities:**
  - Developers need local agents that function efficiently within constrained memory budgets (~16 GB RAM) without requiring enterprise cloud GPUs.

---

## Slide 3: Objectives & Key Innovations

- **Flexible Inference:**
  - Local generation via Ollama, or optional Groq GPT-OSS 20B/120B cloud generation.
  - RAG embeddings remain local via `nomic-embed-text`; cloud mode sends prompts and retrieved code context to Groq.
- **Genuine Local RAG Architecture:**
  - Line-preserving code chunking, metadata tracking, and ChromaDB vector persistence with MD5 deduplication.
- **Two Distinct Agentic Workflows:**
  - **Task 1:** Evidence-based code investigation and root-cause diagnosis.
  - **Task 2:** Automated fix generation with isolated sandbox validation and human approval gates.
- **Defense-in-Depth Security:**
  - Path traversal protection, Docker container sandboxing, non-root execution, and immutable JSON audit logging.

---

## Slide 4: System Architecture

- **High-Level Subsystems:**
  - **Presentation Layer:** Streamlit UI with multi-tab layout, live model switching, and diff visualizer.
  - **Agent Core:** Bounded tool loop with dynamic tool selection, action parsing, and iteration limits.
  - **RAG Subsystem:** File scanner, line chunker, Ollama embedder, and ChromaDB dual collections.
  - **Execution & Sandbox:** Canonical path validator, Docker/subprocess test runner, and automated backup engine.
- **Visual:** Architectural pipeline diagram (UI $\rightarrow$ Agent Loop $\leftrightarrow$ Local Ollama $\leftrightarrow$ Tool Suite $\leftrightarrow$ ChromaDB).

---

## Slide 5: The Local RAG Pipeline

- **Ingestion & Filtering:**
  - Recursively processes `.py`, `.md`, `.txt`, `.toml`, `.json`, `.yaml`.
  - Defensive exclusions: `__pycache__`, `.git`, `.venv`, and files > 1 MB.
- **Line-Preserving Chunking:**
  - Avoids syntax breaks by splitting on line boundaries (800 character window, 200 character overlap).
  - Embeds precise file path, start line, and end line metadata into every chunk.
- **Local Embeddings & Deduplication:**
  - On-device embedding with `nomic-embed-text` (768-dimensional vectors).
  - Deterministic MD5 IDs ensure zero chunk duplication across re-indexing runs.
- **Dual Collections:**
  - `workspace_code` (project source & documentation) vs. `reference_docs` (debugging notes).

---

## Slide 6: Agentic Task 1 — Code Investigation & Debugging

- **Workflow Description:**
  - Receives natural-language symptom descriptions, error messages, or tracebacks.
  - Agent queries workspace using tools: `list_files`, `read_file`, `search_code`, `get_file_metadata`, `search_knowledge_base`.
- **Key Characteristics:**
  - **Empirical Exploration:** Discovers project structure before examining specific files.
  - **Grounded Citations:** Final explanations cite exact file names and line ranges from tool observations.
  - **Calibrated Uncertainty:** Explicitly reports when evidence is insufficient rather than guessing.
  - **Bounded Execution:** Maximum iteration limits prevent infinite loops.

---

## Slide 7: Agentic Task 2 — Automated Fix Generation & Validation

- **End-to-End Remediation Pipeline:**
  1. Investigates root cause using Task 1 tools.
  2. Generates unified diff using `propose_patch` (zero modifications to original workspace).
  3. Deploys fix into a disposable test sandbox (Docker with `--network none` or temporary copy).
  4. Runs pytest validation suite and analyzes real exit codes.
  5. Automatically retries if validation fails (up to configurable `MAX_FIX_ATTEMPTS`).
  6. Displays unified diff preview in the UI for explicit human approval.
  7. Creates timestamped backup in `.codepilot_backups/` before applying approved changes.

---

## Slide 8: Technology Stack & Memory Efficiency

- **Frontend:** Streamlit 1.31+
- **LLM Engine:** Ollama local daemon or optional Groq GPT-OSS cloud API
- **Primary Coding Model:** `qwen2.5-coder:7b` (4.7 GB)
- **Low-Memory Fallbacks:** `llama3.2:3b` (2.0 GB), `llama3.2:1b` (1.3 GB)
- **Embedding Model:** `nomic-embed-text` (274 MB)
- **Vector Database:** ChromaDB 1.5+ (persistent SQLite store)
- **Sandboxing:** Docker Desktop (read-only, 512 MB memory ceiling, dropped capabilities)
- **Resource Footprint:** Operates smoothly within 16 GB DDR4 RAM workstation constraints.

---

## Slide 9: Verification, Testing Results & Demonstration

- **Comprehensive Automated Test Suite:**
  - **Total Tests:** 55 passed in 55.6 seconds.
  - **Unit Tests:** 50 tests verifying config, path traversal guards, line chunker, tool execution, and action parser.
  - **Live Integration Tests:** 5 tests validating live Ollama inference, embedding generation, and vector retrieval.
- **Synthetic Buggy Workspace Evaluation:**
  - Project repository with 7 intentional bugs across mathematical and string modules.
  - Initial baseline: 11 failed, 9 passed out of 20 pytest test cases.
  - Verified agent investigation and patch generation against failing test cases.

---

## Slide 10: Limitations, Future Work & Conclusion

- **Operational Limitations:**
  - Sub-7B models occasionally require regex fallback parsing.
  - Single-file patch application per agent cycle.
  - CPU inference adds minor per-turn latency compared to GPU acceleration.
- **Future Enhancements:**
  - Tree-sitter AST parsing for syntax-aware code chunking.
  - Multi-file patch transactions.
  - GPU-accelerated local execution.
- **Conclusion:**
  - Successfully transformed `llmlocal` into **CodePilot Local**.
  - Met all coursework requirements: 2 genuine agentic workflows, true local RAG, human-in-the-loop safety, and full academic documentation.
