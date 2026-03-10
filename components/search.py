# components/search.py

from tavily import TavilyClient
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# ─────────────────────────────────────
# 1. Initialize Tavily Client
# ─────────────────────────────────────
def get_tavily_client():
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("❌ TAVILY_API_KEY not found in .env file!")
    return TavilyClient(api_key=api_key)


# ─────────────────────────────────────
# 2. Search the Web
# ─────────────────────────────────────
def search_web(query, max_results=5):
    """
    Takes a search query
    Returns top web results as a list of dicts
    """
    client = get_tavily_client()

    response = client.search(
        query=query,
        search_depth="advanced",   # deeper search
        max_results=max_results
    )

    results = []
    for r in response["results"]:
        results.append({
            "title": r["title"],
            "url": r["url"],
            "content": r["content"]
        })

    print(f"✅ Found {len(results)} results for: {query}")
    return results


# ─────────────────────────────────────
# 3. Summarize Search Results with LLM
# ─────────────────────────────────────
def summarize_search_results(query, results):
    """
    Takes raw search results → sends to Llama 3
    Returns a clean research summary
    """
    llm = OllamaLLM(model="llama3", temperature=0.3)

    # Combine all result content into one text block
    combined_results = "\n\n".join([
        f"Source: {r['title']}\nURL: {r['url']}\nContent: {r['content']}"
        for r in results
    ])

    prompt_template = """
    You are a research assistant. Based on the following web search results,
    write a clear and concise research summary about: "{query}"

    Focus on:
    - Key findings and facts
    - Important concepts
    - Any trends or insights

    Web Search Results:
    {results}

    Research Summary:
    """

    prompt = PromptTemplate(
        input_variables=["query", "results"],
        template=prompt_template
    )

    chain = prompt | llm

    summary = chain.invoke({
        "query": query,
        "results": combined_results
    })

    return summary


# ─────────────────────────────────────
# 4. Full Pipeline — Search + Summarize
# ─────────────────────────────────────
def research_topic(query):
    """
    Main function — search web + summarize results
    Returns summary + list of sources
    """
    # Step 1: Search
    results = search_web(query)

    if not results:
        return {
            "summary": "❌ No results found. Try a different query.",
            "sources": []
        }

    # Step 2: Summarize with Llama 3
    summary = summarize_search_results(query, results)

    # Step 3: Return summary + sources
    sources = [{"title": r["title"], "url": r["url"]} for r in results]

    return {
        "summary": summary,
        "sources": sources
    }