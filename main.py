from graph_flow import app

print("🤖 Customer Support Bot (RAG System)")
print("Type 'exit' to quit\n")

while True:
    query = input("Ask your question: ")

    if query.lower() == "exit":
        break

    result = app.invoke({"query": query})

    print("\nAnswer:", result["answer"])