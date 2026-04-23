from langgraph.graph import StateGraph
from retriever import get_retriever
from hitl import human_escalation

retriever = get_retriever()

def process(state):
    query = state["query"]

    docs = retriever.invoke(query)

    # Combine text
    context = " ".join([doc.page_content for doc in docs])

    # ✅ SIMPLE RELEVANCE CHECK
    query_words = query.lower().split()

    match_count = sum(word in context.lower() for word in query_words)

    # If very low match → NOT relevant → HITL
    if match_count < 2:
        return {"answer": human_escalation(query)}

    # Otherwise return answer
    answer = f"\n📄 Based on document:\n{context[:500]}...\n"

    return {"answer": answer}


graph = StateGraph(dict)

graph.add_node("process", process)

graph.set_entry_point("process")
graph.set_finish_point("process")

app = graph.compile()