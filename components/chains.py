# components/chains.py

from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document


# ─────────────────────────────────────
# 1. Load Llama 3 via Ollama
# ─────────────────────────────────────
def get_llm():
    llm = OllamaLLM(
        model="llama3",
        temperature=0.1,
        num_predict=512,
    )
    return llm


# ─────────────────────────────────────
# 2. Summarization Chain (Fixed)
# ─────────────────────────────────────
def summarize_docs(chunks):
    llm = get_llm()

    # ─────────────────────────────────────
    # اختار chunks من الأول والوسط والنهاية
    # ─────────────────────────────────────
    total = len(chunks)
    print(f"✅ Total chunks: {total}")

    selected_chunks = (
        chunks[:30] +                           # أول 8 صفحات
        chunks[total//2:total//2 + 30] +        # وسط الكتاب
        chunks[-30:]                            # آخر 8 صفحات
    )

    # شيل التكرار لو في chunks مشتركة
    seen = set()
    unique_chunks = []
    for chunk in selected_chunks:
        if chunk.page_content not in seen:
            seen.add(chunk.page_content)
            unique_chunks.append(chunk)

    print(f"✅ Selected {len(unique_chunks)} unique chunks for summarization")

    # ─────────────────────────────────────
    # قسّم على مجموعات كل مجموعة 5 chunks
    # ─────────────────────────────────────
    def chunk_groups(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    group_summaries = []
    groups = list(chunk_groups(unique_chunks, 5))
    print(f"✅ Total groups: {len(groups)}")

    for i, group in enumerate(groups):
        combined_text = "\n\n".join([
            doc.page_content for doc in group
        ])

        if not combined_text.strip():
            continue

        prompt = f"""Summarize the following text concisely in 3-5 sentences:

{combined_text}

Summary:"""

        print(f"⏳ Summarizing group {i+1}/{len(groups)}...")
        mini_summary = llm.invoke(prompt)
        group_summaries.append(mini_summary)

    # ─────────────────────────────────────
    # ادمج كل الملخصات في ملخص نهائي
    # ─────────────────────────────────────
    if not group_summaries:
        return "❌ Could not extract text from document."

    all_summaries = "\n\n".join(group_summaries)

    final_prompt = f"""Based on these section summaries from the beginning, 
middle, and end of a document, write one comprehensive final summary 
in clear paragraphs covering the main topics and key points:

{all_summaries}

Final Summary:"""

    print("⏳ Generating final summary...")
    final_summary = llm.invoke(final_prompt)
    return final_summary


# ─────────────────────────────────────
# 3. Q&A Chain
# ─────────────────────────────────────
def answer_question(question, retriever):
    llm = get_llm()

    prompt_template = """
    You are a helpful research assistant.
    Use the following context from the document to answer the question.
    If you don't know the answer from the context, say "I couldn't find 
    this information in the document."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    result = qa_chain.invoke({"query": question})

    return {
        "answer": result["result"],
        "sources": result["source_documents"]
    }