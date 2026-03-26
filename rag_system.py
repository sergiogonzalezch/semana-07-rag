from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pathlib import Path
import os


load_dotenv()

VECTORSTORE_PATH = "vectorstore_index"


def load_all_documents(directory="./docs"):
    """Carga todos los documentos de texto y PDF desde el directorio especificado.
    """
    print(f"\nCargando documentos de {directory}...")
    documents = []

    txt_loader = DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader)
    documents.extend(txt_loader.load())

    for pdf_path in Path(directory).rglob("*.pdf"):
        loader = PyPDFLoader(str(pdf_path))
        documents.extend(loader.load())

    print(f"\n{len(documents)} documentos cargados")
    return documents


def split_documents(documents):
    """Realiza el chunking de los documentos usando RecursiveCharacterTextSplitter."""
    print("\nDividiendo documentos en chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"{len(chunks)} chunks creados")
    return chunks


def get_embeddings():
    """Carga el modelo de embeddings de HuggingFace. Se configura para usar CPU y normalizar los embeddings."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_or_create_vectorstore(chunks):
    """Carga un vector store existente desde disco o crea uno nuevo a partir de los chunks."""
    embeddings = get_embeddings()

    if Path(VECTORSTORE_PATH).exists():
        print("\nCargando vector store existente...")
        vectorstore = FAISS.load_local(
            VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True
        )
        print("\nVector store cargado desde disco")
    else:
        print(f"\nCreando embeddings para {len(chunks)} chunks...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(VECTORSTORE_PATH)
        print("\nVector store creado y guardado en disco")

    return vectorstore


# def test_search(vectorstore):
#     queries = [
#         "¿Cuánto cuesta un café?",
#         "horario de trabajo de los empleados",
#         "¿Cómo funciona la fotosíntesis?",
#     ]
#     for query in queries:
#         results = vectorstore.similarity_search_with_score(query, k=4)
#         print(f"\nBúsqueda: '{query}'")
#         for i, (doc, score) in enumerate(results, 1):
#             source = doc.metadata.get("source", "N/A")
#             preview = doc.page_content[:60].replace("\n", " ")
#             print(f"  {i}. [{score:.3f}] {source}")
#             print(f"     {preview}...")


def get_llm():
    """Configura el modelo de lenguaje de OpenAI usando variables de entorno para la clave API, el nombre del modelo y la URL base."""
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY", "not-needed"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )


def create_qa_chain(vectorstore):
    print("\nCreando QA chain...")

    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )

    system_prompt = (
        "Usa SOLAMENTE el siguiente contexto para responder la pregunta. "
        "Si la información no está en el contexto, responde: "
        "'No tengo esa información en los documentos proporcionados.' "
        "NO inventes información. NO uses tu conocimiento general. "
        "Cuando cites información, menciona de qué parte del contexto proviene.\n\n"
        "Historial de conversación:\n{chat_history}\n\n"
        "Contexto:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(get_llm(), prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("\nSistema RAG listo\n")
    return rag_chain


def ask_question(rag_chain, question, chat_history, show_chunks=False):
    print(f"\nPregunta: {question}\n")

    history_text = ""

    for q, a in chat_history[-5:]:
        history_text += f"Usuario: {q}\nAsistente: {a}\n"

    result = rag_chain.invoke(
        {
            "input": question,
            "chat_history": history_text if history_text else "(sin historial)",
        }
    )

    print(f"\nRespuesta:\n{result['answer']}\n")

    if result.get("context"):
        print("\nFuentes utilizadas:")
        seen_sources = set()
        for doc in result["context"]:
            source = doc.metadata.get("source", "Desconocida")
            if source not in seen_sources:
                seen_sources.add(source)
                print(f"  - {source}")

        if show_chunks:
            print("\nChunks recuperados (debugging):")
            for i, doc in enumerate(result["context"], 1):
                print(f"\n  --- Chunk {i} ---")
                print(f"  Fuente: {doc.metadata.get('source', 'N/A')}")
                print(f"  Texto: {doc.page_content[:150]}...")

    return result["answer"]

def log_chat_history(chat_history, log_file="logs/chat_history.txt"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        for question, answer in chat_history:
            f.write(f"Usuario: {question}\n")
            f.write(f"Asistente: {answer}\n")
            f.write("-" * 40 + "\n")


def main():
    print("🚀 Iniciando sistema RAG...\n")
    documents = load_all_documents("./docs")
    if not documents:
        print("No se encontraron documentos en ./docs")
        return
    chunks = split_documents(documents)
    vectorstore = load_or_create_vectorstore(chunks)
    rag_chain = create_qa_chain(vectorstore)

    chat_history = []

    print("Escribe 'salir' para terminar.\n")
    while True:
        try:
            question = input("Tu pregunta: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n¡Hasta luego!")
            break
        if question.lower() in ["salir", "exit", "quit", ""]:
            print("¡Hasta luego!")
            break
        answer = ask_question(rag_chain, question, chat_history, show_chunks=False)
        chat_history.append((question, answer))
        log_chat_history(chat_history)


if __name__ == "__main__":
    main()
    # docs = load_documents("./docs")
    # chunks = split_documents(docs)
    # vectorstore = create_vector_store(chunks)
    # test_search(vectorstore)


# if __name__ == "__main__":
#     docs = load_documents("./docs")
#     chunks = split_documents(docs)

#     print(f"\n--- Total: {len(docs)} documentos → {len(chunks)} chunks ---\n")

#     for i, chunk in enumerate(chunks[:3]):
#         print(f"Chunk {i+1}:")
#         print(f"  Fuente: {chunk.metadata['source']}")
#         print(f"  Largo: {len(chunk.page_content)} caracteres")
#         print(f"  Texto: {chunk.page_content[:80]}...")
#         print()
