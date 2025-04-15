import getpass
import openai
import os
os.environ["USER_AGENT"] = "RAGtool/1.0 (Python 3.11; Windows 11; custom langchain integration)"
openai.api_key = os.environ["OPENAI_API_KEY"]
if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

os.environ["LANGSMITH_TRACING"] = "true"
#os.environ["LANGSMITH_API_KEY"] = ""


os.environ["LANGSMITH_ENDPOINT"] ="https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"]="RAGtool"
from langchain.chat_models import init_chat_model

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)


import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

# Load and chunk contents of the blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

#response = graph.invoke({"question": "What is Task Decomposition?"})
#response = graph.invoke({"question":getpass.getpass("Enter your question for RAGtool:")})
#response = graph.invoke({"question": "Summarise Chain of Hindsight in 5 sentences"})

try:
    user_question = input("Please enter your question: ")

    if not user_question.strip():
        raise ValueError("No question entered. Please enter a valid question.")

    # Use llm to get the response
    response = graph.invoke({"question":user_question})
    print(response["answer"])
except Exception as e:
    print(f"An error occurred: {e}")

#print(response["answer"])