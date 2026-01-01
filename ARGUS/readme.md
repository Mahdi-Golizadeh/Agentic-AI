# ARGUS: An Agentic RAG Framework for Grounded Scientific Question Answering

This repository implements an **agentic, self-correcting Retrieval-Augmented Generation (RAG) system** designed for **research-grade question answering** over scientific literature.  
The system is specifically instantiated and tested on **knowledge distillation for object detection**, but the architecture is domain-agnostic.

Unlike standard RAG pipelines, this system explicitly models **reasoning steps** such as query rewriting, relevance grading, retry control, and grounded answer generation.

---

## ✨ Key Features

- **Agentic control flow** using LangGraph
- **Self-correcting retrieval** via query rewriting
- **LLM-based relevance grading** (binary, explainable)
- **Strict grounding**: answers are generated *only* from verified relevant documents
- **Hallucination resistance** by construction
- **Research-oriented design**, suitable for academic usage

---

## 🧠 System Architecture

The system is composed of the following agents:

1. **Query Rewriter**
   - Reformulates user queries to improve semantic retrieval
   - Enforces single-sentence, retrieval-optimized outputs

2. **Retriever**
   - Retrieves candidate document chunks from a vector database

3. **Relevance Grader**
   - Uses an LLM to judge whether each retrieved chunk helps answer the query
   - Rejects irrelevant or weakly related content

4. **Retry Controller**
   - Decides whether to retry retrieval with a rewritten query
   - Uses a bounded retry budget

5. **Answer Generator**
   - Produces a final answer using **only the verified relevant chunks**
   - Refuses to answer when evidence is insufficient

### Control Flow

```

User Query
↓
Retrieve Documents
↓
Grade Relevance
↓
Enough Evidence?
├── No → Rewrite Query → Retrieve Again
└── Yes → Generate Grounded Answer

```

---

## 📂 Repository Structure

```

src/
├── agents/
│   ├── query_rewriter.py        # Query optimization agent
│   ├── relevance_grader.py      # Binary relevance classifier
│   ├── answer_generator.py      # Grounded answer generator
│
├── ingestion/
│   ├── ingest.py                # PDF loading and chunking
│   ├── vectorstore.py           # Vector DB construction
│
├── graph/
│   ├── nodes.py                 # LangGraph nodes
│   ├── graph.py                 # Graph definition
│
├── llm/
│   ├── model.py                 # Singleton LLM loader

````

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install torch transformers langchain langgraph faiss-cpu bitsandbytes
````

> A GPU is recommended. The system supports 4-bit quantized LLMs.

---

### 2. Ingest Documents

Place your PDFs (e.g., KD-for-object-detection papers) in the data directory, then run:

```python
from src.ingestion.ingest import load_papers, chunk_documents
from src.ingestion.vectorstore import build_vectorstore

docs = load_papers()
chunks = chunk_documents(docs)
vectordb = build_vectorstore(chunks)
```

---

### 3. Run the Agentic RAG System

```python
initial_state = {
    "query": "How does knowledge distillation improve small object detection?",
    "retry_count": 0
}

final_state = graph.invoke(initial_state)

print("Final Query:", final_state["query"])
print("Retries:", final_state["retry_count"])
print("Relevant Chunks:", len(final_state["relevant_docs"]))
print("\nAnswer:\n", final_state["answer"])
```

---

## 🧪 Example Behavior

### Covered Question

```
Query:
How does feature-level knowledge distillation improve object detection?

Answer:
Feature-level knowledge distillation improves object detection by transferring
semantic information from intermediate layers of a teacher model to a student,
enabling better representation learning for small objects...
```

### Unsupported Question

```
Query:
Does knowledge distillation always improve inference speed?

Answer:
The provided documents do not contain enough information to answer the question.
```

✔ No hallucination
✔ Explicit uncertainty handling

---

## 🔒 Design Principles

* **Grounding over fluency**
* **Explicit reasoning steps**
* **No silent failures**
* **Deterministic control flow**
* **Explainable decisions**

This system intentionally prioritizes **correctness and transparency** over aggressive answer generation.

---

## 🎓 Intended Use Cases

* Literature review assistance
* Academic question answering
* Research exploration with strict grounding
* Studying agentic RAG architectures
* PhD-level system prototyping

---

## 📌 Notes on Relevance Strictness

The relevance grader is intentionally conservative.
If no document explicitly or implicitly supports a query, the system will refuse to answer.

This behavior can be relaxed by modifying the relevance grading prompt, depending on the desired trade-off between strictness and coverage.

---

## 📜 License

This project is intended for research and educational use.

---

## ✍️ Author

Developed as part of a research-oriented exploration of **agentic RAG systems** and **knowledge distillation in object detection**.

```