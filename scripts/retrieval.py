import chromadb

from pgrag.embeddings.llama_embeddings import embed_text


def main():
    client = chromadb.PersistentClient(path="data/chroma")

    collection = client.get_collection(name="project_gorgon")

    query = input("Question: ")

    query_embedding = embed_text(query)

    results = collection.query(
        query_embeddings=[query_embedding], n_results=10, include=["documents", "distances"]
    )

    print("\nResults:\n")

    for i, doc in enumerate(results["documents"][0]):
        print("=" * 60)
        print(f"Result {i + 1}")
        print(doc[:1000])

    print("Distance:", results["distances"][0][i])


if __name__ == "__main__":
    main()
