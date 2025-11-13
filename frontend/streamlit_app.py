
import streamlit as st
import plotly.express as px
import pandas as pd
import requests
from dotenv import load_dotenv
import os

# load .env
load_dotenv(".env")
backend_url = os.getenv("BACKEND_URL") 
# print(backend_url)

# Page config
# st.set_page_config(page_title="Frontend App", layout="wide")

# Page config
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown(
'''
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .source-box {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            border-left: 3px solid #1f77b4;
        }
        .metric-card {
            background-color: #ffffff;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
''', unsafe_allow_html=True)

# Upload Document
def upload_document(file):
    '''Upload a document to the backend and return the response.'''
    try:
        with st.spinner("Uploading document..."):
            files = {"file": (file.name, file, file.type)}
            response = requests.post(f"{backend_url}/upload", files=files)
            response.raise_for_status()
            return response.json()
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error: {http_err.response.status_code} - {http_err.response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Connection error: Unable to reach the backend.")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
    return None

# Ask Question
def ask_question(question, document_ids=None, top_k=4):
    '''Send a question to the backend and return the response.'''
    try:
        payload = {
            "question": question,
            "document_ids": document_ids,
            "top_k": top_k,
            "include_sources": True
        }
        with st.spinner("Asking the backend..."):
            response = requests.post(f"{backend_url}/ask", json=payload)
            response.raise_for_status()
            return response.json()
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error: {http_err.response.status_code} - {http_err.response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Connection error: Unable to reach the backend.")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
    return None

# Get Documents
def get_documents():
    '''Fetch the list of available documents from the backend.'''
    try:
        with st.spinner("Fetching documents..."):
            response = requests.get(f"{backend_url}/documents")
            if response.status_code == 404:
                return []  # Treat 404 as empty list
            response.raise_for_status()
            return response.json()
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error: {http_err.response.status_code} - {http_err.response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Connection error: Unable to reach the backend.")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
    return None

# Delete document(s)
def delete_document(document_id):
    '''Delete a document by ID from the backend.'''
    try:
        with st.spinner(f"Deleting document: {document_id}"):
            response = requests.delete(f"{backend_url}/documents/{document_id}")
            response.raise_for_status()
            return True
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error: {http_err.response.status_code} - {http_err.response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Connection error: Unable to reach the backend.")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
    return False


def check_backend_health():
    '''Check if backend is running'''
    try:
        response = requests.get(f"{backend_url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False



def main():
    # Header
    st.markdown('<h1 class="main-header">📚 RAG Document Assistant</h1>', unsafe_allow_html=True)

    # Check backend health
    if not check_backend_health():
        st.error("⚠️ Backend server is not running. Please start it with: `uvicorn app.main:app --reload`")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Top-K slider
        top_k = st.slider(
            "Number of sources to retrieve",
            min_value=1,
            max_value=10,
            value=4,
            help="Higher values provide more context but may include less relevant information"
        )

        st.divider()

        # Document management
        st.header("📄 Documents")

        docs_response = get_documents()
        if docs_response and docs_response.get('documents'):
            documents = docs_response['documents']
            st.success(f"✅ {len(documents)} document(s) indexed")

            # Document selection
            selected_docs = st.multiselect(
                "Filter by documents",
                options=[doc['document_id'] for doc in documents],
                format_func=lambda x: next(
                    (doc['filename'] for doc in documents if doc['document_id'] == x),
                    x
                ),
                help="Leave empty to search all documents"
            )

            # Show document details
            with st.expander("View document details"):
                for doc in documents:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.text(f"📄 {doc['filename']}")
                        st.caption(f"Pages: {doc['pages']} | Chunks: {doc['chunks']}")
                    with col2:
                        if st.button("🗑️", key=f"delete_{doc['document_id']}"):
                            if delete_document(doc['document_id']):
                                st.success("Deleted!")
                                st.rerun()
        else:
            st.info("No documents uploaded yet")
            selected_docs = None

        st.divider()

        # About
        st.header("ℹ️ About")
        st.markdown('''
        This is a RAG (Retrieval-Augmented Generation) system that:
        - 📤 Accepts PDF, DOCX, and TXT files
        - 🔍 Finds relevant information
        - 💬 Answers questions with citations
        - 🎯 Provides source references
        ''')

    # Main content area - Two tabs
    tab1, tab2 = st.tabs(["💬 Ask Questions", "📤 Upload Documents"])

    # Tab 1: Question Answering
    with tab1:
        st.header("Ask Questions About Your Document(s)")

        # Check if documents exist
        if not docs_response or not docs_response.get('documents'):
            st.warning("⚠️ Please upload document(s) using the 'Upload Documents' tab")
        else:
            # Question input
            question = st.text_input(
                "Enter your question:",
                placeholder="What is the main topic discussed in the document?",
                key="question_input"
            )

            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                ask_button = st.button("🔍 Ask Question", type="primary", use_container_width=True)
            with col2:
                if st.button("🔄 Clear", use_container_width=True):
                    st.session_state.question_input = ""
                    st.rerun()

            # Process question
            if ask_button and question:
                with st.spinner("🤔 Thinking..."):
                    result = ask_question(
                        question,
                        document_ids=selected_docs if selected_docs else None,
                        top_k=top_k
                    )

                if result:
                    # Display answer
                    st.success("✅ Answer Generated")

                    # Answer box
                    st.markdown("### 💡 Answer")
                    st.markdown(f'''
                    <div style='background-color: #e8f4f8; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4;'>
                        {result['answer']}
                    </div>
                    ''', unsafe_allow_html=True)

                    # Metadata
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("⏱️ Processing Time", f"{result['processing_time']:.2f}s")
                    with col2:
                        st.metric("🤖 Model", result['model_used'])
                    with col3:
                        st.metric("📚 Sources Used", len(result['sources']))

                    # Sources
                    if result.get('sources'):
                        st.markdown("### 📖 Sources")
                        for i, source in enumerate(result['sources'], 1):
                            with st.expander(
                                f"Source {i}: {source['document_name']} "
                                f"(Page {source['page_number']}) - "
                                f"Relevance: {source['relevance_score']:.1%}"
                            ):
                                st.text(source['chunk_text'])
            elif ask_button:
                st.warning("Please enter a question")

    # Tab 2: Document Upload
    with tab2:
        st.header("Upload New Documents")

        # Upload interface
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'docx', 'txt'],
            help="Supported formats: PDF, DOCX, TXT (Max 10MB)"
        )

        if uploaded_file:
            # Display file info
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"📄 **File:** {uploaded_file.name}")
            with col2:
                file_size_mb = uploaded_file.size / (1024 * 1024)
                st.info(f"💾 **Size:** {file_size_mb:.2f} MB")

            # Upload button
            if st.button("📤 Upload and Process", type="primary", use_container_width=True):
                with st.spinner("🔄 Processing document..."):
                    # Show progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    status_text.text("📄 Uploading document...")
                    progress_bar.progress(25)

                    result = upload_document(uploaded_file)

                    if result:
                        status_text.text("✂️ Chunking text...")
                        progress_bar.progress(50)
                        time.sleep(0.5)

                        status_text.text("🧮 Generating embeddings...")
                        progress_bar.progress(75)
                        time.sleep(0.5)

                        status_text.text("💾 Storing in vector database...")
                        progress_bar.progress(100)

                        # Success message
                        st.success("✅ Document processed successfully!")

                        # Display results
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📄 Pages", result['pages'])
                        with col2:
                            st.metric("✂️ Chunks", result['chunks'])
                        with col3:
                            st.metric("🆔 ID", result['document_id'][:8] + "...")

                        st.info("💡 You can now ask questions about this document in the 'Ask Questions' tab")

                        # Clear progress
                        time.sleep(1)
                        progress_bar.empty()
                        status_text.empty()
        else:
            # Instructions
            st.markdown('''
            ### 📝 Instructions

            1. **Click** the file uploader above
            2. **Select** a PDF, DOCX, or TXT file (max 10MB)
            3. **Click** 'Upload and Process'
            4. **Wait** for processing to complete
            5. **Go** to 'Ask Questions' tab to query your document

            ### 💡 Tips
            - Documents are automatically chunked for optimal retrieval
            - Each chunk includes page number for source tracking
            - Multiple documents can be uploaded and queried together
            - Use the sidebar to filter which documents to search
            ''')

    # Footer
    st.divider()
    st.markdown('''
    <div style='text-align: center; color: #666; padding: 1rem;'>
        Built with ❤️ using Streamlit, FastAPI, LangChain, and ChromaDB.<br>
        RAG Document Assistant v1.0.0
    </div>
    ''', unsafe_allow_html=True)


if __name__ == "__main__":
    main()




