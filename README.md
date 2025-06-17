# Falcon LLM App 🚀

A multimodal chat + reasoning app powered by Falcon LLM.

### Features
- 🔤 Text chat with reasoning
- 🖼️ Text + Image (multimodal) input
- 🧠 Optional FAISS-based memory

## 🔧 Local Dev

```bash
pip install -r app/requirements.txt
uvicorn app.main:app --reload
