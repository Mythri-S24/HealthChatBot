# 🏥 MediCare AI — Domain-Specific Healthcare Chatbot with RAG

A domain-specific conversational AI chatbot built for **Healthcare FAQs**, powered by **Retrieval-Augmented Generation (RAG)** and an open-source LLM (**FLAN-T5**), with a polished **Gradio** web interface.

---

## 📌 What This Project Does

MediCare AI lets users ask natural-language health questions and receive concise, context-grounded answers. It combines a lightweight retrieval engine with a generative language model — no external API or paid service required.

**Example queries the bot can handle:**
- "What are the symptoms of fever?"
- "How is diabetes managed?"
- "Tell me about COVID-19 treatment."
- "What should I do for a minor burn?"

---

## 🏗️ Architecture Overview

```
User Query
    │
    ▼
┌─────────────────────────┐
│   DocumentRetriever     │  ← TF-IDF + Cosine Similarity
│   (finds top-k docs)    │
└─────────────┬───────────┘
              │  top relevant document(s)
              ▼
┌─────────────────────────┐
│    LLMGenerator         │  ← google/flan-t5-base
│  (builds prompt + gen)  │
└─────────────┬───────────┘
              │  generated answer
              ▼
┌─────────────────────────┐
│    Gradio UI            │  ← Landing page + Chat page
└─────────────────────────┘
```

The system follows a classic **RAG pipeline**:
1. The user's query is vectorized.
2. The retriever finds the most semantically relevant document(s) from the knowledge base.
3. The best-matching document is injected into a structured prompt.
4. FLAN-T5 generates a 2–3 sentence answer grounded in that document.

---

## 🧠 Core Components

### 1. Knowledge Base
A hardcoded list of 8 healthcare FAQ documents covering:

| Topic | Description |
|---|---|
| Fever | Temperature thresholds, causes, when to seek care |
| Diabetes | Type 1 vs Type 2, symptoms, management |
| Hypertension | Blood pressure thresholds, lifestyle, medication |
| COVID-19 | Symptoms, vaccination, antivirals |
| Mental Health | Common disorders, therapies, early intervention |
| Nutrition | WHO guidelines, balanced diet |
| Exercise | WHO recommendations, cardiovascular benefits |
| First Aid – Burns | Cooling methods, what to avoid |

---

### 2. `DocumentRetriever`
Retrieves the top-k most relevant documents for a user query using **TF-IDF vectorization** and **cosine similarity**.

```python
retriever = DocumentRetriever(KNOWLEDGE_BASE, num_docs=2)
results = retriever.retrieve("What causes high blood pressure?")
# → Returns top matching documents with relevance scores
```

**Key design choices:**
- `TfidfVectorizer` with `ngram_range=(1, 2)` captures both unigrams and bigrams.
- Only documents with a similarity score `> 0` are returned.

---

### 3. `LLMGenerator`
Uses **`google/flan-t5-base`** (a free, open-source seq2seq model) to generate answers from retrieved context.

```python
generator = LLMGenerator()
answer = generator.generate(query="What is diabetes?", context_docs=[...])
```

**Prompt structure:**
```
You are a healthcare assistant.
Answer ONLY using the information below.

Information:
<best retrieved document content>

Question: <user query>

Give a clear 2-3 sentence answer.
```

Generation uses **beam search** (`num_beams=5`, `do_sample=False`) for deterministic, factual responses.

---

### 4. `HealthcareChatbot`
Orchestrates the full RAG pipeline end-to-end.

```python
bot = HealthcareChatbot(num_docs=2, temperature=0.7, max_tokens=150)
bot.chat("I have a fever. What should I do?")
```

---

### 5. Gradio Frontend (`app.py`)
A two-page web UI built with **Gradio Blocks**:

**Landing Page** — Full-screen hero section with background image, glassmorphism card, and an "Enter System" button.

**Chat Page** — Dashboard-style header, chat window, text input, and action buttons (Send, Clear, Back).

```python
demo.launch()  # Starts the local Gradio server
```

---

## 📁 Project Structure

```
healthcare_project/
├── backend/
│   └── app.py          # DocumentRetriever, LLMGenerator, HealthcareChatbot
└── frontend/
    └── app.py          # Gradio UI

app.py                  # Standalone all-in-one version (backend + frontend)
backend.py              # Standalone backend-only version
```

> The notebook iterates toward a clean standalone `app.py` that contains everything in one file — recommended for Colab or quick deployment.

---

## ⚙️ Installation

```bash
pip install scikit-learn numpy requests
pip install transformers torch sentencepiece
pip install gradio
```

---

## 🚀 Running the App

**Option 1 — Standalone (recommended):**
```bash
python app.py
```

**Option 2 — Modular:**
```bash
# Run backend first (loads model)
python healthcare_project/backend/app.py

# Then run frontend
python healthcare_project/frontend/app.py
```

After launch, Gradio will print a local URL (e.g., `http://127.0.0.1:7860`). Open it in your browser.

---

## 🔍 How RAG Works Here (Simple Explanation)

Traditional chatbots either hallucinate facts or require a massive model. RAG solves this by:

1. **Retrieving** — finding the right document before generating.
2. **Augmenting** — feeding that document as context into the prompt.
3. **Generating** — letting the LLM answer *only from what it was given*.

This keeps answers factual, controllable, and lightweight — ideal for domain-specific applications like healthcare.

---

## ⚠️ Disclaimer

> This chatbot is for **educational and informational purposes only**. It does not replace professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns.

---

## 🛠️ Tech Stack

| Component | Library / Model |
|---|---|
| Retrieval | `scikit-learn` (TF-IDF, cosine similarity) |
| Language Model | `google/flan-t5-base` via HuggingFace Transformers |
| UI Framework | `gradio` |
| Numerical ops | `numpy` |
| Runtime | Python 3.x, Google Colab / local |
