# MediQuery — RAG Medical Information Assistant

A Retrieval-Augmented Generation (RAG) assistant that answers general health questions by retrieving relevant passages from a curated knowledge base and grounding the language model's answer in that retrieved text.

## Why RAG here

A plain LLM can hallucinate facts or answer from outdated training data, and it can't point to where an answer came from. RAG fixes this by retrieving trusted, current text at query time and forcing the model to answer only from that context — which matters a lot for anything health-related.

## Architecture

```
User question
     │
     ▼
Embedding model (all-MiniLM-L6-v2)
     │
     ▼
FAISS vector search  ──►  top-k relevant chunks
     │
     ▼
Prompt = instruction + retrieved context + question
     │
     ▼
Generator (flan-t5-base)  ──►  answer + sources + disclaimer
```

A safety layer runs before generation: emergency-keyword detection routes potential emergencies to a fixed "contact emergency services" response instead of a generated answer, and every normal answer ends with a disclaimer that this is general information, not medical advice.

## Tech stack

- `sentence-transformers` — embeddings (`all-MiniLM-L6-v2`)
- `faiss-cpu` — vector similarity search
- `transformers` — generation (`google/flan-t5-base`)
- `langchain` — text chunking (`RecursiveCharacterTextSplitter`)
- `gradio` — optional chat UI

## Run it

Open `MediQuery_RAG_Assistant.ipynb` in Google Colab or Jupyter and run the cells top to bottom. The default models need no API key and run on a free Colab CPU/GPU.

To use a stronger model instead of `flan-t5-base`, only the generator cell (`Step 6`) and `mediquery_answer` need to change — the retrieval pipeline is unaffected, so it's a drop-in swap for an API-based LLM (OpenAI, Anthropic, or a hosted open-weight model).

## Current limitations

- The knowledge base is a small, hand-written demo set of ~12 general health topics — not a vetted medical corpus.
- No re-ranking, conversational memory, or answer-faithfulness evaluation yet (see Future Improvements in the notebook).
- Not clinically validated. Not a diagnostic tool. Not a substitute for professional medical advice.
