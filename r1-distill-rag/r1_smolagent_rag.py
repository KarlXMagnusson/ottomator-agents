import os
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
import gradio as gr

load_dotenv()

# Load environment variables
reasoning_model_id = os.getenv("REASONING_MODEL_ID", "mistral")  # Default to Mistral
tool_model_id = os.getenv("TOOL_MODEL_ID", "mistral")

# Load Ollama LLM
llm = Ollama(model=reasoning_model_id)

# Initialize vector store with Ollama embeddings
db_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectordb = Chroma(persist_directory=db_dir, embedding_function=embeddings)

# Conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create ConversationalRetrievalChain (Replacing Reasoner)
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectordb.as_retriever(),
    memory=memory
)

def rag_with_reasoner(user_query: str):
    """
    Performs a RAG search and answers a query using an LLM.
    """
    response = rag_chain.invoke({"question": user_query})
    return response["answer"]

# Gradio Interface
def chat_interface(user_query):
    return rag_with_reasoner(user_query)

iface = gr.Interface(fn=chat_interface, inputs="text", outputs="text", title="RAG Chatbot")

if __name__ == "__main__":
    iface.launch()