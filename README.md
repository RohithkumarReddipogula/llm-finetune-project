# LLM Fine-Tuning with QLoRA
### Fine-tuned TinyLlama 1.1B on NLP and RAG domain knowledge

<p align="center">
<img src="https://img.shields.io/badge/ Model-HuggingFace-yellow?style=for-the-badge" />
<img src="https://img.shields.io/badge/Method-QLoRA-green?style=for-the-badge" />
<img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python" />
<img src="https://img.shields.io/badge/GPU-T4%20Free%20Colab-orange?style=for-the-badge" />
</p>

<p align="center">
<b>Fine-tuned a Large Language Model on NLP and RAG domain knowledge
using QLoRA — training only 0.089% of parameters on a free GPU.</b>
</p>

<p align="center">
<b>Published Model:</b> <a href="https://huggingface.co/Rohith2026/nlp-rag-expert">huggingface.co/Rohith2026/nlp-rag-expert</a>
</p>

---

## What I Built

A domain-specific fine-tuned LLM that specialises in answering NLP and RAG
questions — my exact area of expertise from my MSc thesis on hybrid retrieval.

The model was fine-tuned using **QLoRA** (Quantised Low-Rank Adaptation) —
the industry-standard technique for efficient LLM fine-tuning used at every
serious AI company.

---

## What Is QLoRA and Why Does It Matter

Standard fine-tuning requires updating 100% of model parameters.
For a 1.1B model that means storing and updating 1.1 billion values —
impossible on a free 15GB T4 GPU.

QLoRA solves this elegantly:

    Full model (frozen, 4-bit) ──► Only 0.089% of parameters trained
           ↓                              ↓
    4x memory reduction            Small LoRA matrices added
           ↓                              ↓
    Fits on free T4 GPU            Comparable to full fine-tuning

This is the same technique used to fine-tune 7B to 70B models in production.

---

##  Training Results

| Metric | Value |
|--------|-------|
| Base model | TinyLlama 1.1B Chat |
| Fine-tuning method | QLoRA (4-bit NF4 quantisation) |
| LoRA rank | 16 |
| Total parameters | 1.1 Billion |
| Parameters trained | 0.089% only |
| Training examples | 10 NLP/RAG instruction pairs |
| Starting loss | 2.47 |
| Final loss | 0.89 |
| Epochs | 3 |
| Training time | Under 10 minutes |
| Hardware | NVIDIA T4 GPU (Google Colab free tier) |

---

##  Tech Stack

| Tool | Purpose | Why I chose it |
|------|---------|----------------|
| **Unsloth** | QLoRA training library | 2x faster + 60% less memory than standard HuggingFace |
| **HuggingFace PEFT** | LoRA adapter management | Industry standard — in 70% of ML job descriptions |
| **TRL SFTTrainer** | Supervised fine-tuning | Used at HuggingFace and major AI labs |
| **bitsandbytes** | 4-bit quantisation | Essential for running LLMs on limited hardware |
| **HuggingFace Hub** | Model publishing | Public proof of fine-tuning skills |
| **Google Colab T4** | Free GPU | Accessible to anyone — no cloud spend needed |

---

## Training Domain

The model was fine-tuned to answer expert questions about:

- Retrieval-Augmented Generation (RAG) systems
- BM25 vs dense retrieval — differences and tradeoffs
- FAISS vector databases and index types
- LoRA and QLoRA fine-tuning techniques
- LLM evaluation frameworks (RAGAS, MRR, Recall@K)
- Semantic search and dense embeddings
- Hybrid retrieval strategies

This domain was chosen because it is my thesis area — fine-tuning on
what you know deeply produces better training examples.

---

## My Honest Observations

With 10 training examples on a 1.1B model, the experiment demonstrated
the complete QLoRA workflow end-to-end. The model showed partial domain
adaptation — the technique works, but scale matters.

**What I learned:**
- Larger datasets (1000+ examples) produce much stronger domain specialisation
- Bigger models (7B+) respond better to fine-tuning
- Free Colab T4 constraints limited this experiment intentionally
- The workflow is identical at scale — only the dataset size and model change

This is the honest engineering insight: understanding the limits of an
approach is as valuable as making it work.

---

## How to Use This Model

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    model = PeftModel.from_pretrained(base, "Rohith2026/nlp-rag-expert")
    tokenizer = AutoTokenizer.from_pretrained("Rohith2026/nlp-rag-expert")

    prompt = "### Instruction:\nWhat is RAG?\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=200)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

---

## My Full AI Portfolio

| Project | Description | Live |
|---------|-------------|------|
| [Hybrid RAG System](https://github.com/RohithkumarReddipogula/AI-Powered-Rag-System) | BM25 + dense embeddings · 93% Recall@10 · 8.84M passages · MSc thesis | [Demo](https://rohith2026-hybrid-rag-demo.hf.space) · [API](https://rohith2026-hybrid-rag-api.hf.space/docs) |
| [AI Agent System](https://github.com/RohithkumarReddipogula/ai-agent-project) | ReAct agent · LangGraph · 3 tools · web search + calculator + RAG | [Demo](https://rohith2026-ai-agent-react.hf.space) |
| LLM Fine-Tuning (this) | QLoRA · TinyLlama 1.1B · NLP/RAG domain · published model | [Model](https://huggingface.co/Rohith2026/nlp-rag-expert) |

---

## Author

**Rohith Kumar Reddipogula**
MSc Data Science — University of Europe for Applied Sciences, Berlin

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin)](https://linkedin.com/in/rohith-kumar-reddipogula-a6692030b)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github)](https://github.com/RohithkumarReddipogula)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Profile-yellow?style=flat)](https://huggingface.co/Rohith2026)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?style=flat&logo=gmail)](mailto:rohithkumar336699@gmail.com)

---
