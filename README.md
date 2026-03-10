# 🔬 AI Research Assistant

An intelligent research assistant powered by local AI (Llama 3) that helps you 
summarize documents, answer questions, and search the web for research topics.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2.16-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38-red)
![Ollama](https://img.shields.io/badge/Ollama-Llama3-orange)

---

## ✨ Features

- 📄 **Upload & Summarize** — Upload PDF or TXT files and get an instant AI-powered summary
- ❓ **Document Q&A** — Ask questions about your document using RAG (Retrieval Augmented Generation)
- 🌐 **Web Research** — Search the web and get AI-summarized research results with sources

---

## 🛠️ Tech Stack

| Component | Tool |
|-----------|------|
| 🧠 LLM | Ollama (Llama 3) — runs 100% locally |
| 📐 Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| 🗄️ Vector Store | FAISS |
| 🌐 Web Search | Tavily API |
| ⛓️ Orchestration | LangChain |
| 🎨 UI | Streamlit |

---

## 🏗️ Project Structure
```
ai-research-assistant/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not uploaded to GitHub)
├── .gitignore              # Git ignore rules
│
├── components/
│   ├── loader.py           # PDF & TXT file loader + text splitter
│   ├── embedder.py         # HuggingFace embeddings + FAISS vector store
│   ├── chains.py           # LangChain chains (summarize + Q&A)
│   └── search.py           # Tavily web search + summarization
│
└── vectorstore/            # FAISS index storage
```

---

## 🔄 How It Works
```
Upload PDF/TXT
      ↓
LangChain reads & splits into chunks
      ↓
HuggingFace converts chunks → vectors
      ↓
Vectors saved in FAISS
      ↓
  ┌───┴───┐
  ↓       ↓
Summarize  Q&A
  ↓       ↓
Llama 3 generates response
      ↓
Displayed in Streamlit UI
```

---

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/gempooo-29/ai-research-assistant.git
cd ai-research-assistant
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup environment variables
Create a `.env` file in the root folder:
```
TAVILY_API_KEY=your_tavily_api_key_here
```
Get your free Tavily API key at 👉 https://tavily.com

### 5. Install & run Ollama
```bash
# Download Ollama from https://ollama.com
ollama pull llama3
ollama serve
```

### 6. Run the app
```bash
streamlit run app.py
```

Open your browser at 👉 `http://localhost:8501`

---

## 📖 How To Use

### 📄 Tab 1 — Upload & Summarize
1. Click **Browse files** and upload a PDF or TXT file
2. Click **⚡ Process & Summarize**
3. Wait for the AI to generate a summary
4. Download the summary as TXT if needed

### ❓ Tab 2 — Ask Document
1. First upload and process a document in Tab 1
2. Type your question in the chat box
3. Get AI-powered answers based on your document
4. View source chunks used to generate the answer

### 🌐 Tab 3 — Web Research
1. Type any research topic
2. Click **🔍 Research**
3. Get an AI-summarized research report
4. View sources and download the report

---

## ⚙️ Requirements

- Python 3.10+
- Ollama installed with Llama 3 pulled
- Tavily API key (free tier: 1000 searches/month)
- 8GB RAM minimum (for running Llama 3 locally)

---

## 🔒 Privacy & Security

- ✅ Llama 3 runs **100% locally** — your documents never leave your machine
- ✅ Only web search queries are sent to Tavily API
- ✅ No data is stored permanently

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👨‍💻 Author

Made with ❤️ by [gempooo-29](https://github.com/gempooo-29)
