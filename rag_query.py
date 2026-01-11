import chromadb
import ollama

CHROMA_DIR = "data/chroma"
COLLECTION_NAME = "tunisia_archaeo"

# Initialisation de ChromaDB
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(COLLECTION_NAME)

def retrieve_context(query, k=10):
    """
    Récupère les k chunks les plus pertinents depuis ChromaDB.
    Priorise les chunks contenant les mots-clés de la question.
    """
    results = collection.query(
        query_texts=[query],
        n_results=k
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    context = ""
    formatted_sources = []
    seen_titles = set()

    # Extraire mots clés simples pour priorité
    keywords = [w.lower() for w in query.split() if len(w) > 3]

    prioritized_chunks = []
    normal_chunks = []

    for i, (chunk, meta) in enumerate(zip(docs, metas)):
        title = meta.get("title", "Unknown")
        if title in seen_titles:
            continue
        seen_titles.add(title)

        chunk_lower = chunk.lower()
        if any(word in chunk_lower for word in keywords):
            prioritized_chunks.append((chunk, title))
        else:
            normal_chunks.append((chunk, title))

    # Ajouter d'abord les chunks prioritaires
    all_chunks = prioritized_chunks + normal_chunks

    for i, (chunk, title) in enumerate(all_chunks):
        context += f"[Chunk {i+1} from {title}]:\n{chunk}\n"
        formatted_sources.append(f"- {title} ({meta.get('source', 'Unknown')})")

    return context, formatted_sources

def build_prompt(question, context):
    """
    Construit un prompt renforcé pour obtenir des réponses courtes et directes en français.
    """
    return f"""
You are an expert on Tunisian archaeological sites.

Answer the question **based only on the provided context**.
Give a **short, direct answer in French** (one sentence only, without explanation).
If the answer is not found in the context, say:
"Je ne dispose pas d'information fiable sur ce point."

### Contexte :
{context}

### Question :
{question}

### Réponse (courte et factuelle) :
"""

def ask_rag(question):
    """
    Récupère le contexte depuis ChromaDB, construit le prompt et interroge Ollama.
    """
    context, sources = retrieve_context(question)
    prompt = build_prompt(question, context)

    response = ollama.chat(
        model="gemma3:1b",
        messages=[{"role": "user", "content": prompt}]
    )

    answer_text = response.get("message", {}).get("content", "Erreur : pas de réponse")

    return answer_text.strip(), sources


if __name__ == "__main__":
    print("✅ RAG prêt à l'emploi avec gemma3:1b et réponses directes !")
    while True:
        q = input("\n🔍 Question : ")
        answer, sources = ask_rag(q)

        print("\n📌 Réponse :")
        print(answer)

        print("\n📚 Sources :")
        for s in sources:
            print(s)
