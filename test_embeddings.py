from langchain_huggingface import HuggingFaceEmbeddings

print("Cargando modelo de embeddings (primera vez puede tardar ~30 seg)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

vector = embeddings.embed_query("¿Cuánto cuesta un café en Aurora?")
print(f"Embedding creado: {len(vector)} dimensiones")
print(f"Primeros 5 valores: {vector[:5]}")