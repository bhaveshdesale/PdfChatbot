import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
import os
from htmlTemplates import css, bot_template, user_template

# Set Streamlit page configuration
st.set_page_config(page_title="Chat with Multiple PDFs", page_icon="📚", layout="wide")

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:  # Only add non-empty text
                text += extracted_text
    return text

def get_text_chunks(text):
    """Split text into manageable chunks for processing."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1500,  # Adjusted chunk size for faster processing
        chunk_overlap=300,  # Adjusted overlap for context
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    st.write("Number of text chunks:", len(chunks))  # Debugging
    return chunks

def get_vectorstore(text_chunks):
    """Create a vector store from text chunks using HuggingFace embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def handle_userinput(user_question):
    """Handle user questions and display conversation history."""
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def get_conversation_chain(vectorstore):
    """Create a conversation chain using a HuggingFace language model."""
    hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_api_token:
        st.error("Hugging Face API token not found. Please set the HUGGINGFACEHUB_API_TOKEN environment variable.")
        return None

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-small",  # Smaller model for faster responses
        model_kwargs={"temperature": 0.5, "max_length": 512},  # Reduced max_length
        huggingfacehub_api_token=hf_api_token
    )

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def set_css():
    """Set custom CSS styles for the Streamlit app."""
    st.markdown("""
        <style>
        .main-header {
            font-size: 36px;
            color: #3c6e71;
            text-align: center;
            padding-top: 10px;
        }
        .pdf-upload {
            text-align: center;
            margin-bottom: 20px;
        }
        .question-box {
            margin: 20px 0;
        }
        .question-input {
            font-size: 18px;
            padding: 10px;
        }
        .process-button {
            background-color: #3c6e71;
            color: white;
            font-size: 18px;
            padding: 10px;
            border-radius: 5px;
            cursor: pointer;
        }
        .process-button:hover {
            background-color: #285e5a;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    """Main function to run the Streamlit app."""
    load_dotenv()
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None    

    set_css()

    st.markdown('<div class="main-header">Chat with Multiple PDFs 📚</div>', unsafe_allow_html=True)
    
    user_question = st.text_input("Ask a question about your documents:", placeholder="Type your question here...")
    if user_question and st.session_state.conversation:
        handle_userinput(user_question)
    elif user_question and not st.session_state.conversation:
        st.warning("Please upload and process PDFs first.")

    with st.sidebar:
        st.subheader("📂 Upload Your Documents")
        
        pdf_docs = st.file_uploader("Upload your PDFs here:", accept_multiple_files=True, type=["pdf"])

        if st.button("Process", key="process_button"):
            if pdf_docs:
                with st.spinner("Processing"):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)

                    if vectorstore:
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        if st.session_state.conversation:
                            st.success("PDFs uploaded and processed successfully!")
                        else:
                            st.error("Failed to create conversation chain.")
            else:
                st.error("Please upload at least one PDF.")

if __name__ == "__main__":
    main()



# #Code for Local Model

# import streamlit as st
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# from langchain.llms.base import LLM
# from typing import List, Optional
# import os
# from htmlTemplates import css, bot_template, user_template

# # Set Streamlit page configuration at the top
# st.set_page_config(page_title="Chat with Multiple PDFs", page_icon="📚", layout="wide")

# class LocalHuggingFaceLLM(LLM):
#     """Custom LLM class to use the local Hugging Face model."""
    
#     model_name: str  # Define this as a pydantic field
#     tokenizer: AutoTokenizer = None  # Define tokenizer as a class field
#     model: AutoModelForSeq2SeqLM = None  # Define model as a class field

#     def __init__(self, model_name: str):
#         # Call the parent constructor to satisfy pydantic's requirements
#         super().__init__(model_name=model_name)
#         self.model_name = model_name
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

#     def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
#         inputs = self.tokenizer.encode(prompt, return_tensors="pt")
#         outputs = self.model.generate(inputs, max_length=512, num_beams=5, early_stopping=True)
#         return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

#     @property
#     def _identifying_params(self):
#         return {"model_name": self.model_name}

#     @property
#     def _llm_type(self) -> str:
#         return "custom"

# def get_pdf_text(pdf_docs):
#     """Extract text from uploaded PDF documents."""
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text() or ""
#     return text

# def get_text_chunks(text):
#     """Split text into manageable chunks for processing."""
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vectorstore(text_chunks):
#     """Create a vector store from text chunks using HuggingFace embeddings."""
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorstore

# def handle_userinput(user_question):
#     """Handle user questions and display conversation history.""" 
#     response = st.session_state.conversation({'question': user_question})
#     st.session_state.chat_history = response['chat_history']
    
#     for i, message in enumerate(st.session_state.chat_history):
#         if i % 2 == 0:
#             st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
#         else:
#             st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

# def get_conversation_chain(vectorstore):
#     """Create a conversation chain using a local HuggingFace language model."""
    
#     # Use a smaller model to reduce memory usage
#     local_llm = LocalHuggingFaceLLM(model_name="google/flan-t5-base")

#     memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

#     # Create the conversation chain using the local model and the vectorstore
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=local_llm,
#         retriever=vectorstore.as_retriever(),
#         memory=memory
#     )
#     return conversation_chain

# def set_css():
#     """Set custom CSS styles for the Streamlit app."""
#     st.markdown(""" 
#         <style>
#         .main-header {
#             font-size: 36px;
#             color: #3c6e71;
#             text-align: center;
#             padding-top: 10px;
#         }
#         .pdf-upload {
#             text-align: center;
#             margin-bottom: 20px;
#         }
#         .question-box {
#             margin: 20px 0;
#         }
#         .question-input {
#             font-size: 18px;
#             padding: 10px;
#         }
#         .process-button {
#             background-color: #3c6e71;
#             color: white;
#             font-size: 18px;
#             padding: 10px;
#             border-radius: 5px;
#             cursor: pointer;
#         }
#         .process-button:hover {
#             background-color: #285e5a;
#         }
#         </style>
#     """, unsafe_allow_html=True)

# def main():
#     """Main function to run the Streamlit app."""
#     load_dotenv()  # Load environment variables from .env file
#     st.write(css, unsafe_allow_html=True)
    
#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None
        
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = None    

#     set_css()

#     st.markdown('<div class="main-header">Chat with Multiple PDFs 📚</div>', unsafe_allow_html=True)
    
#     st.markdown("<div class='question-box'>", unsafe_allow_html=True)
#     user_question = st.text_input("Ask a question about your documents:", placeholder="Type your question here...")
#     if user_question and st.session_state.conversation:
#         handle_userinput(user_question)
#     elif user_question and not st.session_state.conversation:
#         st.warning("Please upload and process PDFs first.")
#     st.markdown("</div>", unsafe_allow_html=True)

#     with st.sidebar:
#         st.subheader("📂 Upload Your Documents")
        
#         st.markdown("<div class='pdf-upload'>", unsafe_allow_html=True)
#         pdf_docs = st.file_uploader("Upload your PDFs here:", accept_multiple_files=True, type=["pdf"])
#         st.markdown("</div>", unsafe_allow_html=True)

#         if st.button("Process", key="process_button"):
#             if pdf_docs:
#                 with st.spinner("Processing"):
#                     # Get PDF text
#                     raw_text = get_pdf_text(pdf_docs)

#                     # Get the text chunks
#                     text_chunks = get_text_chunks(raw_text)

#                     # Create vector store
#                     vectorstore = get_vectorstore(text_chunks)

#                     if vectorstore:
#                         # Create conversation chain
#                         st.session_state.conversation = get_conversation_chain(vectorstore)
#                         if st.session_state.conversation:
#                             st.success("PDFs uploaded and processed successfully!")
#                         else:
#                             st.error("Failed to create conversation chain.")
#             else:
#                 st.error("Please upload at least one PDF.")

#     st.markdown("---")
#     st.markdown("### How to Use the App:")
#     st.markdown("""
#         - **Step 1:** Upload one or more PDF documents using the sidebar.
#         - **Step 2:** Click on "Process" to analyze the documents.
#         - **Step 3:** Ask questions about the content of the PDFs.
#     """)

# if __name__ == "__main__":
#     main()