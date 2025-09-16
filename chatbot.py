import ollama
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- Configuration ---
DATA_PATH = "./rawdata"
QDRANT_HOST = "localhost"
QDRANT_COLLECTION_NAME = "aoe2_docs"
EMBEDDING_MODEL = "mxbai-embed-large"
# LLM_MODEL = "gemma3:12b-it-qat"
LLM_MODEL = "gemma3:1b"

def ingest_data():
    """
    Loads data from the source directory, splits it into chunks,
    creates embeddings, and stores them in Qdrant.
    """
    print("Starting data ingestion...")

    # 1. Load Documents
    loader = DirectoryLoader(DATA_PATH, glob="**/*.md", show_progress=True)
    documents = loader.load()
    if not documents:
        print("No documents found. Please add text files to the 'data' directory.")
        return

    print(f"Loaded {len(documents)} documents.")

    # 2. Split Documents into Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    # 3. Create Embeddings and Store in Qdrant
    print(f"Creating embeddings with '{EMBEDDING_MODEL}' and storing in Qdrant...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # This command connects to Qdrant, creates the collection if it doesn't exist,
    # generates embeddings for the chunks, and stores them.
    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=f"http://{QDRANT_HOST}:6333",
        collection_name=QDRANT_COLLECTION_NAME,
        force_recreate=True, # Use True to start fresh each time, False to append
    )

    print("Data ingestion complete!")

def main():
    """
    Sets up the RAG chain and starts an interactive question-answering loop.
    """
    # Connect to the existing Qdrant vector store
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    client = QdrantClient(url=f"http://{QDRANT_HOST}:6333")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION_NAME,
        embedding=embeddings
    )
    retriever = vector_store.as_retriever()

    # Define the prompt template
    template = """
    You are an expert on the game Age of Empires 2.
    Answer the question based only on the following context.
    If you don't know the answer from the context provided, just say that you don't know.

    Context:
    {context}

    Question:
    {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Initialize the Ollama LLM
    llm = ollama.Client()

    # Create the RAG chain
    # This defines the flow:
    # 1. Retrieve relevant context based on the question.
    # 2. Format the prompt with the context and question.
    # 3. Pass the formatted prompt to the LLM.
    # 4. Parse the output.
    def ollama_llm(prompt_value):
        # Convert ChatPromptValue to string
        prompt_str = prompt_value.to_string()
        response = llm.chat(
            model=LLM_MODEL,
            messages=[{'role': 'user', 'content': prompt_str}],
            stream=False
        )
        return response['message']['content']

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | ollama_llm
        | StrOutputParser()
    )

    print("\nChatbot is ready! Type 'exit' to quit.")
    while True:
        question = input("\nAsk a question about AoE2: ")
        if question.lower() == 'exit':
            break

        print("\nThinking...")
        answer = rag_chain.invoke(question)
        print("\nAnswer:")
        print(answer)

if __name__ == "__main__":
    ingest_data()
    main()
