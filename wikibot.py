import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import warnings
warnings.filterwarnings("ignore")
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os
from dotenv import load_dotenv
from vector_embeddings import vector
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Load environment variables from .env file
load_dotenv()

data_directory = os.path.join(os.path.dirname(__file__), "data")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# st.secrets["huggingface_api_token"] # Don't forget to add your hugging face token
# Ensure the data directory exists
if not os.path.exists(data_directory):
    os.makedirs(data_directory)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Load the vector store from disk
vector_store = Chroma(embedding_function=embedding_model, persist_directory=data_directory)

# Initialize the Hugging Face Hub LLM
hf_hub_llm = HuggingFaceHub(
     repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    # repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    model_kwargs={"temperature": 1, "max_new_tokens":1024},
)

prompt_template = """
You are an intelligent Wikipedia-based chatbot. Your task is to respond to user queries using only the information 
scraped from the Wikipedia page. Provide clear, concise, and direct answers that are relevant to the question, based on 
the content of the Wikipedia article. Avoid adding any external information or unrelated content.

Article Context: {context}

User Question: {question}

Answer:
"""

custom_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

rag_chain = RetrievalQA.from_chain_type(
    llm=hf_hub_llm, 
    chain_type="stuff", 
    retriever=vector_store.as_retriever(top_k=3),  # retriever is set to fetch top 3 results
    chain_type_kwargs={"prompt": custom_prompt})

def get_response(question):
    result = rag_chain({"query": question})
    response_text = result["result"]
    answer_start = response_text.find("Answer:") + len("Answer:")
    answer = response_text[answer_start:].strip()
    return answer

# Streamlit app
# Remove whitespace from the top of the page and sidebar
st.markdown(
    """
    <style>
        .appview-container .main .block-container {{
            padding-top: {padding_top}rem;
            padding-bottom: {padding_bottom}rem;
        }}
    </style>
    """.format(padding_top=1, padding_bottom=1),
    unsafe_allow_html=True,
)

# Header for the Wikipedia bot
st.markdown("""
    <h3 style='text-align: left; color: white; padding-top: 35px; border-bottom: 3px solid blue;'>
        Discover Knowledge with WikiBot 🌐📚
    </h3>""", unsafe_allow_html=True)

# Sidebar message
side_bar_message = """
Hi! 👋 I'm WikiBot, your companion for exploring Wikipedia. 
Here's how I can help you:
1. **Summaries of Topics** 📖
2. **In-depth Information** 🔍
3. **Related Articles** 🌐
4. **Quick Facts** 💡

Just ask me about any topic, and I’ll help you browse Wikipedia!
"""

with st.sidebar:
    st.title('WikiBot🤖: Your Wikipedia Companion')
    st.markdown(side_bar_message)

# Initial message for main chat area
initial_message = """
    Hello! I'm WikiBot🤖
    What's on your mind?:\n 
     📘 "Cricket?"\n 
     📘 "Machine Learning?"\n 
     📘 "Mathematics?"\n 
     📘 "Donald Trump?"\n 
     📘 "Jackie Chan?" 
"""
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": initial_message}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#def clear_chat_history():
#    st.session_state.messages.clear()
#    st.session_state.messages = [{"role": "assistant", "content": initial_message}]
#st.button('Clear Chat', on_click=clear_chat_history)

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
            st.markdown(prompt)
    if len(st.session_state.messages) == 2:
        with st.chat_message("assistant"):
            with st.spinner("Hold on, I'm fetching the Data from WIKIPEDIA..."):
                name = st.session_state.messages[1]["content"]
                formatted_name = "_".join([word.capitalize() for word in name.split()])
                vector(formatted_name)
                response = f"Data Fetched! What do you want to know about {prompt}?"
                placeholder = st.empty()
                full_response = response
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)
        
# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant" and len(st.session_state.messages) > 2:
    with st.chat_message("assistant"):
        with st.spinner("Hold on, Thinking..."):
            name = st.session_state.messages[1]["content"]
            formatted_name = "_".join([word.capitalize() for word in name.split()])
            if not prompt.endswith('?'):
                prompt += '?'
            prompt1 = f"{prompt} ({formatted_name})"
            response = get_response(prompt1)
            placeholder = st.empty()
            full_response = response  # Directly use the response
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)

if st.button("End Chat and Exit"):
    st.write("Thank you for chatting! Goodbye!")
    st.stop()
    sys.exit()