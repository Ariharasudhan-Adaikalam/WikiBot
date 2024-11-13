from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os
import shutil
from dotenv import load_dotenv
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore")
from wiki_content import get_wiki

# Load environment variables from .env file
load_dotenv()

data_directory = os.path.join(os.path.dirname(__file__), "data")
if os.path.exists(data_directory):
    shutil.rmtree(data_directory)
    print(f"Deleted directory: {data_directory}")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# st.secrets["huggingface_api_token"] # Don't forget to add your hugging face token
# Ensure the data directory exists
if not os.path.exists(data_directory):
    os.makedirs(data_directory)

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Chroma vector store
vector_store = Chroma(embedding_function=embedding_model, persist_directory=data_directory)

def vector(search):
    full_page_content= get_wiki(search)
    documents = [Document(page_content=full_page_content)] 

    # Split the text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    texts = text_splitter.split_documents(documents)
    # Convert texts to embeddings
    try:
        embeddings = embedding_model.embed_documents([doc.page_content for doc in texts])
    except Exception as e:
        print(f"Error creating vector embeddings: {e}")

    # Add documents to the Chroma vector store
    try:
        vector_store.add_documents(documents=texts)
        vector_store.persist()
    except Exception as e:
        print(f"Error adding data to vector store: {e}")