# 🤖 Agentic AI — Open-Source Autonomous Systems (No LangChain, No Paid LLMs)

This repository is a growing collection of **agentic AI projects built entirely with free, open-source small language models (SLMs)** that run on **Google Colab Free T4 GPU**, without using:

❌ LangChain  
❌ Paid APIs (OpenAI, Anthropic, Gemini)  
❌ Proprietary cloud services  

Instead, every project uses:

✔ Pure Python  
✔ Lightweight open-source models (Qwen2.5, Phi-3, Gemma, Mistral, etc.)  
✔ 4-bit quantization for Colab T4 efficiency  
✔ A custom minimal agent framework  
✔ Open-source tools (Pandas, PyPDF, Requests, BeautifulSoup, etc.)

The goal is to demonstrate **practical agent engineering** under realistic resource constraints.

---

# 📚 Table of Contents

- [About This Repository](#about-this-repository)
- [What Makes This Repo Different](#what-makes-this-repo-different)
- [Projects](#projects)
  - [1. Data-Cleaning Agent](#1-data-cleaning-agent)
  - [2. Research Assistant Agent](#2-research-assistant-agent)
  - [3. Task-Planning Agent](#3-task-planning-agent)
  - [4. Multi-Agent PDF Inspector](#4-multi-agent-pdf-inspector)

---

# 🧠 About This Repository

**Agentic AI** is designed as a long-term portfolio that shows how to build **real autonomous agents using small, efficient, free models** instead of relying on cloud APIs or large proprietary LLMs.

This repo reflects the philosophy:

> *“Powerful agentic systems do not require huge LLMs — only well-designed tools, structure, and small models used effectively.”*

The entire codebase:
- runs on free Google Colab  
- uses small models (1B–7B) in 4-bit quantization  
- is transparent, modular, and reproducible  
- avoids heavy frameworks  

---

# What Makes This Repo Different

### ✔ Runs Fully Local (Colab)  
Every project is tested on a free T4 GPU.

### ✔ Uses Only Open-Source Small Models  
Examples: Qwen2.5-3B, Phi-3 3.8B, Gemma-2B, Mistral-7B.

### ✔ No LangChain  
We implement our **own** agent loops, tool-use logic, memory, and message passing.

### ✔ Minimal Dependencies  
Everything is based on standard Python libraries.

### ✔ Focus on Real-World Agentic Systems  
Not toy examples — practical, end-to-end workflows.

---

# 🚀 Projects

Each project lives in its own folder with its own README, code, and examples.

---

## 1️⃣ Data-Cleaning Agent
  
**Status:** Planned

A Python-first agent that:
- understands user instructions with a small LLM  
- applies transformations with Pandas  
- documents each operation  
- outputs a cleaned dataset  

---

## 2️⃣ Research Assistant Agent
 
**Status:** Planned

An agent that:
- generates search queries  
- scrapes the web  
- extracts text  
- summarizes and produces a structured report  

---

## 3️⃣ Task-Planning Agent
  
**Status:** Planned

A hybrid planner:
- SLM extracts goals  
- Python planner generates steps  
- tools execute steps  
- agent monitors progress  

---

## 4️⃣ Multi-Agent PDF Inspector
 
**Status:** Planned

A modular multi-agent system with roles:
- Extractor Agent  
- Analyzer Agent  
- Summarizer Agent  
- Coordinator  

---
