# app.py

import streamlit as st
from components.loader import load_file
from components.embedder import create_vectorstore, get_retriever
from components.chains import summarize_docs, answer_question
from components.search import research_topic

# ─────────────────────────────────────
# Page Config
# ─────────────────────────────────────
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide"
)

# ─────────────────────────────────────
# Header
# ─────────────────────────────────────
st.title("🔬 AI Research Assistant")
st.markdown("Powered by **Llama 3** • **LangChain** • **HuggingFace**")
st.divider()

# ─────────────────────────────────────
# Session State (memory between actions)
# ─────────────────────────────────────
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ─────────────────────────────────────
# Layout — 3 Tabs
# ─────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📄 Upload & Summarize",
    "❓ Ask Document",
    "🌐 Web Research"
])


# ═════════════════════════════════════
# TAB 1 — Upload & Summarize
# ═════════════════════════════════════
with tab1:
    st.header("📄 Upload Your Document")
    st.markdown("Upload a **PDF** or **TXT** file to get an instant summary.")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt"],
        help="Supports PDF and TXT files"
    )

    if uploaded_file is not None:
        st.success(f"✅ File uploaded: **{uploaded_file.name}**")

        col1, col2 = st.columns(2)

        # Process file button
        with col1:
            if st.button("⚡ Process & Summarize", use_container_width=True):
                with st.spinner("📖 Reading document..."):
                    chunks = load_file(uploaded_file)

                with st.spinner("🧠 Creating vector store..."):
                    st.session_state.vectorstore = create_vectorstore(chunks)

                with st.spinner("✍️ Generating summary..."):
                    st.session_state.summary = summarize_docs(chunks)

                st.success("✅ Done! Summary ready.")

        # Clear button
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.vectorstore = None
                st.session_state.summary = None
                st.session_state.chat_history = []
                st.rerun()

    # Show summary
    if st.session_state.summary:
        st.divider()
        st.subheader("📝 Document Summary")
        st.markdown(st.session_state.summary)

        # Download summary
        st.download_button(
            label="⬇️ Download Summary",
            data=st.session_state.summary,
            file_name="summary.txt",
            mime="text/plain"
        )


# ═════════════════════════════════════
# TAB 2 — Q&A on Document
# ═════════════════════════════════════
with tab2:
    st.header("❓ Ask Questions About Your Document")

    if st.session_state.vectorstore is None:
        st.warning("⚠️ Please upload and process a document in Tab 1 first!")
    else:
        st.success("✅ Document loaded and ready for questions!")

        # Chat history display
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        question = st.chat_input("Ask anything about your document...")

        if question:
            # Show user message
            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })
            with st.chat_message("user"):
                st.markdown(question)

            # Get answer
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    retriever = get_retriever(st.session_state.vectorstore)
                    result = answer_question(question, retriever)
                    answer = result["answer"]
                    sources = result["sources"]

                st.markdown(answer)

                # Show source chunks
                if sources:
                    with st.expander("📚 View Source Chunks"):
                        for i, doc in enumerate(sources):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.markdown(doc.page_content)
                            st.divider()

            # Save assistant message
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer
            })


# ═════════════════════════════════════
# TAB 3 — Web Research
# ═════════════════════════════════════
with tab3:
    st.header("🌐 Web Research")
    st.markdown("Search the web and get an **AI-powered research summary**.")

    search_query = st.text_input(
        "Enter a research topic",
        placeholder="e.g. Latest trends in large language models 2024"
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        search_btn = st.button("🔍 Research", use_container_width=True)

    if search_btn and search_query:
        with st.spinner("🌐 Searching the web..."):
            results = research_topic(search_query)

        st.divider()
        st.subheader("📋 Research Summary")
        st.markdown(results["summary"])

        # Show sources
        if results["sources"]:
            st.divider()
            st.subheader("🔗 Sources")
            for source in results["sources"]:
                st.markdown(f"- [{source['title']}]({source['url']})")

        # Download research
        st.download_button(
            label="⬇️ Download Research",
            data=results["summary"],
            file_name="research.txt",
            mime="text/plain"
        )

    elif search_btn and not search_query:
        st.warning("⚠️ Please enter a research topic first!")