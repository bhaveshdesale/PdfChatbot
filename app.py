#pip install streamlit pypdf2 langchain python-dotenv faiss-c
#pu openai huggingface_hub
#streamlit used to create graphical user interface
#and pypdf2 to interact with our pdf and langchain to interact with our language model, faiss-cpu as our vector store

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter



def get_pdf_text(pdf_docs):
        text=""
        for pdf in pdf_docs:
            pdf_reader=PdfReader(pdf)
            for page in pdf_reader.pages:
                text+=page.extract_text()
        return text  
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Inline CSS for custom styling
def set_css():
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
    load_dotenv()
   
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon="📚", layout="wide")

    set_css()

    st.markdown('<div class="main-header">Chat with Multiple PDFs 📚</div>', unsafe_allow_html=True)
    
    st.markdown("<div class='question-box'>", unsafe_allow_html=True)
    user_question = st.text_input("Ask a question about your documents:", placeholder="Type your question here...")
    st.markdown("</div>", unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("📂 Upload Your Documents")
        
        
        st.markdown("<div class='pdf-upload'>", unsafe_allow_html=True)
        pdf_docs = st.file_uploader("Upload your PDFs here:", accept_multiple_files=True, type=["pdf"])
        st.markdown("</div>", unsafe_allow_html=True)

        
        if st.button("Process", key="process_button"):
            with st.spinner("Processing"):
                #get pdf text
                raw_text=get_pdf_text(pdf_docs)
                #st.write(raw_text)
              

                #get the text chunks
                text_chunks=get_text_chunks(raw_text)
                st.write(text_chunks)



                #create vector store
            if pdf_docs:
                st.success("PDFs uploaded and processed successfully!")
            else:
                st.error("Please upload at least one PDF.")

    
    st.markdown("---")
    st.markdown("### How to Use the App:")
    st.markdown("""
        - **Step 1:** Upload one or more PDF documents using the sidebar.
        - **Step 2:** Ask questions about the content of the PDFs.
        - **Step 3:** The system will analyze the documents and provide answers.
    """)


if __name__ == "__main__":
    main()
