# 🚀 NAVIGATOR: Neural Agentic Vectorized Intelligent Generator And Thoughtful Orchestrator for Retrieval

A next-generation Agentic RAG (Retrieval-Augmented Generation) system leveraging Google Gemini 2.0 Flash, Qdrant vector storage, and Agno (formerly phidata) for agent orchestration. NAVIGATOR is designed for intelligent document and web content processing, advanced query rewriting, and seamless fallback to web search, delivering comprehensive, context-aware AI responses.

**GitHub Repository:**  
https://github.com/shruti-01-narwat/NAVIGATOR

---

## 🌟 Key Features

- **Document & Web Content Processing**
  - Upload and process PDF documents
  - Extract and chunk web page content
  - Embed and store vectors in Qdrant Cloud

- **Intelligent Querying**
  - Automatic query rewriting for optimal retrieval
  - RAG-based similarity search with thresholding
  - Web search fallback with Exa AI integration
  - Source attribution for transparency

- **Agentic Orchestration**
  - Agno Agent framework for modular orchestration
  - Context-aware chat history management
  - Query reformulation and reasoning

- **Modern Interface**
  - Streamlit-powered interactive UI
  - Easy API key management and configuration

---

## 🗂️ File Structure

```
NAVIGATOR/
│
├── agentic_rag_gemini.py        # Main Streamlit app
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
---

## ⚡ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/shruti-01-narwat/NAVIGATOR.git
cd NAVIGATOR
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up API Keys

Create a `.env` file or use the Streamlit sidebar to enter:

- **Google Gemini API Key**  
  Get from [Google AI Studio](https://aistudio.google.com/apikey)

- **Qdrant Cloud Credentials**  
  Get from [Qdrant Cloud](https://cloud.qdrant.io/)

- **Exa AI API Key (Optional)**  
  Get from [Exa AI](https://exa.ai)

Example `.env`:
```
GOOGLE_API_KEY=your-google-api-key
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_URL=https://xxx-xxx.cloud.qdrant.io
EXA_API_KEY=your-exa-api-key
```

### 4. Run the Application

```bash
streamlit run agentic_rag_gemini.py
```

---

## 🛠️ Usage

1. **Configure API keys** in the sidebar.
2. **Upload PDFs** or **enter URLs** for web content.
3. **Ask questions** in the chat interface.
4. **View rewritten queries, sources, and web search results**.
5. **Manage chat history** and configure web search domains as needed.

---

## 🚀 Potential Future Use Cases

- **Enterprise Knowledge Management:**  
  Integrate with internal document stores, wikis, and knowledge bases for instant, context-aware answers.

- **Academic Research Assistant:**  
  Summarize, cross-reference, and answer questions from large collections of research papers.

- **Legal & Compliance Search:**  
  Rapidly retrieve and attribute information from legal documents, contracts, and regulations.

- **Customer Support Automation:**  
  Power intelligent chatbots that can reference both internal docs and the web.

- **Healthcare Information Retrieval:**  
  Securely process and answer queries from medical literature and patient documents.

- **Personal Knowledge Navigator:**  
  Organize, search, and interact with your personal notes, files, and bookmarks.

---

## 🤝 Contributing

Pull requests and feature suggestions are welcome! Please open an issue to discuss your ideas.

---

## 📄 License

MIT License

---

**NAVIGATOR**: Neural Agentic Vectorized Intelligent Generator And Thoughtful Orchestrator for Retrieval  
Empowering intelligent, context-aware information retrieval for everyone.
