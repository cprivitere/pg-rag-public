from pgrag.rag.pipeline import ask


def main():

    question = input("Question: ")

    result = ask(question)

    print("\nAnswer:")
    print(result["answer"])

    print(
        "\nRerank: %s"
        % ("cross-encoder (:8082)" if result.get("rerank_used") else "lexical fallback")
    )

    print("\nSources:")

    for source in result["sources"]:
        print(source["id"], source["metadata"], source["distance"])

    print("\nRetrieved documents:")

    for document in result["documents"]:
        print("\n---")
        print(document[:500])


if __name__ == "__main__":
    main()
