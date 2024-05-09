import streamlit as st
from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import tempfile
import os

OPENAI_API_KEY = st.secrets.openai_api

def get_pdf_text(pdf_docs):
    text = ""
    # Save the uploaded PDF file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_docs.read())
        temp_path = temp_file.name

    try:
        # Load the PDF using PDFMinerLoader
        pdf_reader = PDFMinerLoader(temp_path)
        text = pdf_reader.load()

    finally:
        # Remove the temporary file
        os.remove(temp_path)

    return text

def get_text_chunks(text):
    text = str(text)  # Ensure that text is a string
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, request_timeout=120)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store


st.title("📝 Chatbot - SABIC Materials Q & A ") 

with st.sidebar:
    st.write("""Document Name : SABIC Materials Information \n 
    Document Context: The context is derived from the material data avaialable at the SABIC official page.""")

if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Mention your queries!"}
    ]


# url= "http://172.178.125.127/rts/api/v1/services/sabic_sitedata/rag_based_openai_context_prompt"

url = "http://4.227.155.46/rts/api/v1/services/newupdated/pinecone_version_streamlit"
username = 'soothsayeranalytics'
password = 'Analytics@2024'


@st.cache_resource
def get_refs(pdf_docs,prompt):
    text=get_pdf_text(pdf_docs)
    chunks=get_text_chunks(text)
    vs=get_vector_store(chunks)
    ret=vs.as_retriever()
    results=ret.invoke(prompt)
    info=''
    for result in results:
        info+=result.page_content+' '
    return info

with st.sidebar:
    pdf_docs=st.file_uploader("Upload Files", type=["pdf"])
        
if prompt :=st.text_input("How can i help you today?",placeholder="Your query here"):
    results=get_refs(pdf_docs,prompt)
    prompt=f"Provide the citations and elucidate the concepts of:{prompt}.Include detailed information from relevant sections and sub-sections to ensure a comprehensive response utilizing the given information.The infomation is:{results}"
    st.session_state.messages.append({"role": "user", "content": prompt})
    inputs={ "data":[{"prompt":prompt}]}


# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = requests.post(url, auth=(username, password),json=inputs)
            response_dict = json.loads(response.text)
            # Extracting the content of the "new_col" key
            response2 = response_dict['data'][0]['new_col']
            st.write(response2)
            message = {"role": "assistant", "content": response2}
            st.session_state.messages.append(message)
