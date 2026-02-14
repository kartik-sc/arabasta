---

# 🏺 CodeArchaeologist

<p align="center">
<img src="[https://img.shields.io/badge/CodeArchaeologist-v0.1.0--alpha-blue?style=for-the-badge&logo=gitbook&logoColor=white](https://www.google.com/search?q=https://img.shields.io/badge/CodeArchaeologist-v0.1.0--alpha-blue%3Fstyle%3Dfor-the-badge%26logo%3Dgitbook%26logoColor%3Dwhite)" />
<img src="[https://img.shields.io/badge/Build-Passing-success?style=for-the-badge&logo=github-actions&logoColor=white](https://www.google.com/search?q=https://img.shields.io/badge/Build-Passing-success%3Fstyle%3Dfor-the-badge%26logo%3Dgithub-actions%26logoColor%3Dwhite)" />
<img src="[https://img.shields.io/badge/Security-OWASP--Compliant-red?style=for-the-badge&logo=anchor&logoColor=white](https://www.google.com/search?q=https://img.shields.io/badge/Security-OWASP--Compliant-red%3Fstyle%3Dfor-the-badge%26logo%3Danchor%26logoColor%3Dwhite)" />
</p>

<p align="center">
<b>Stop digging. Start discovering.</b>




<i>The AI-powered "Memory Engine" for legacy codebases and tribal knowledge.</i>
</p>

<p align="center">
<img src="[https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white](https://www.google.com/search?q=https://img.shields.io/badge/Python-3.10%2B-3776AB%3Fstyle%3Dflat-square%26logo%3Dpython%26logoColor%3Dwhite)" />
<img src="[https://img.shields.io/badge/TypeScript-007ACC?style=flat-square&logo=typescript&logoColor=white](https://www.google.com/search?q=https://img.shields.io/badge/TypeScript-007ACC%3Fstyle%3Dflat-square%26logo%3Dtypescript%26logoColor%3Dwhite)" />
<img src="[https://img.shields.io/badge/Neo4j-Graph-008CC1?style=flat-square&logo=neo4j&logoColor=white](https://www.google.com/search?q=https://img.shields.io/badge/Neo4j-Graph-008CC1%3Fstyle%3Dflat-square%26logo%3Dneo4j%26logoColor%3Dwhite)" />
<img src="[https://img.shields.io/badge/ChromaDB-Vector-lightgrey?style=flat-square](https://www.google.com/search?q=https://img.shields.io/badge/ChromaDB-Vector-lightgrey%3Fstyle%3Dflat-square)" />
<img src="[https://img.shields.io/badge/Claude--3.5-Intelligence-7C4DFF?style=flat-square&logo=anthropic&logoColor=white](https://www.google.com/search?q=https://img.shields.io/badge/Claude--3.5-Intelligence-7C4DFF%3Fstyle%3Dflat-square%26logo%3Danthropic%26logoColor%3Dwhite)" />
</p>

---

## 💡 The Problem

Software decays. When senior developers leave, they take **"Tribal Knowledge"** with them—the *why* behind the weird workarounds and the *how* of the legacy spaghetti. New developers spend **60-70% of their time** just trying to understand existing code before writing a single new line.

## 🚀 The Solution

**CodeArchaeologist** is a living map of your engineering soul. It uses state-of-the-art **Contrastive Learning** to understand the **intent** of your code, not just the syntax, while actively "interviewing" developers to capture undocumented decisions.

### 🗺️ System Architecture

```mermaid
graph TD
    A[Legacy Repo] -->|Tree-sitter| B(Multi-Modal Parsing)
    B --> C{SimCLR Encoder}
    C -->|768-dim Vector| D[(ChromaDB)]
    C -->|Relationships| E[(Neo4j Graph)]
    F[Senior Dev] -- AI Interview --> G((Tribal Knowledge Capture))
    G --> E
    D & E --> H[Onboarding Accelerator]
    D & E --> I[Semantic Search]

```

---

## 🛠️ Key Features

| Section | Description | Status |
| --- | --- | --- |
| 🏺 **The Dig Site** | Auto-scans repos, detects languages, and builds dependency graphs. | `RELEASED` |
| 🧠 **The Brain** | Contrastive Learning Engine (SimCLR + GraphCodeBERT). | `BETA` |
| 🗣️ **The Interviewer** | AI chatbot that extracts knowledge from devs during PRs. | `STABLE` |
| 📊 **The Relics** | Tech Debt Tracker and Knowledge Graph visualization. | `WIP` |

---

## 🧬 Deep Tech: The ML Engine

CodeArchaeologist doesn't just look at text; it looks at *semantic structure*.

<details>
<summary><b>📐 How Contrastive Learning Works (For Technical Judges)</b></summary>

We use a **Multi-View Representation** of code. For every snippet, we generate:

1. **The Text View:** Raw source code.
2. **The Structural View:** Abstract Syntax Tree (AST).
3. **The Data Flow View:** How variables move (DFG).

**The Loss Function:**
We minimize the  (Normalized Temperature-scaled Cross Entropy) loss to maximize agreement between different views of the same logic:

This ensures that a "Sort" algorithm in Python and a "Sort" algorithm in C++ map to the same vector space, enabling **Cross-Language Semantic Search**.

</details>

<details>
<summary><b>🕸️ Knowledge Graph Schema</b></summary>

Our Neo4j layer connects the dots:

* `(Developer)-[:AUTHORED]->(Commit)`
* `(Commit)-[:MODIFIED]->(Function)`
* `(Decision)-[:EXPLAINS]->(Function)`
* `(Incident)-[:ROOT_CAUSE_IN]->(File)`

</details>

---

## 📈 Impact (The "Why")

* **Reduce Onboarding Time:** From weeks to days with personalized learning paths.
* **Zero Knowledge Leak:** Institutional memory stays in the repo, not in vanished Slack histories.
* **Smarter Refactoring:** Identify duplicate logic across different microservices instantly.

---

## 🏆 Hackathon Roadmap

* [x] Multi-language Tree-sitter integration (50+ languages)
* [x] SimCLR Contrastive Embedding Pipeline
* [x] AI Interviewer Chatbot (Claude 3.5 Sonnet Integration)
* [ ] VS Code Extension "Archaeologist Lens"
* [ ] Real-time "Fragile Code" alerts via Git Hooks
