import streamlit as st
from constants import *
from modules import get_retriever, get_citations, get_rag,get_query_rephrased
import tempfile
from langchain_community.document_loaders import PyPDFLoader
import traceback


def main():
    try:
        st.title("*Self help* QA Bot")
        st.write(""" Please upload the books with .pdf format and proceed with the QnA
        """)
        # get pdfs to text
        uploaded_files = st.file_uploader("Choose a PDF file", accept_multiple_files=True, type=".pdf")
        documents = []
        if uploaded_files:
            # st.write("files uploaded")
            for uploaded_file in uploaded_files:
                if uploaded_file is not None:
                    try:
                        st.write("using tempfile for ", uploaded_file.name)
                        # using temporary files to store the file and get the path
                        # https://docs.python.org/3/library/tempfile.html ref
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                            temp_file.write(uploaded_file.read())
                            temp_file_path = temp_file.name
                        # st.write("filename:", uploaded_file.name)
                        # loading file in the pdfloader
                        # st.write("loading file in the pdfloader")
                        pdfloader = PyPDFLoader(temp_file_path)
                        document = pdfloader.load()
                        # st.write(document)
                        # temp_file , temp_file_path = None

                        # Extract text content from each document
                        for doc in document:
                            documents.append(doc)
                            
                    except Exception as e:
                        st.error(f"An error occurred while processing the file {uploaded_file.name}: {e}")
                        st.error(traceback.format_exc())
            
            # Join all document texts into a single string
            # all_docs_text = " ".join(documents)
            # st.write("All docs text is extracted")      

                    # Initialize the session state for query history if it doesn't exist
            if 'query_history' not in st.session_state:
                st.session_state.query_history = []
            user_input = st.text_input("Enter your Query ")
            if st.button("Process"):
                retriever = get_retriever(documents)
                # st.info("retriever created")
                rephrased_query = get_query_rephrased(user_input)
                # st.info("rephrased query")
                relevant_docs, content, source,page = get_citations(rephrased_query,retriever)
                # st.write("relevant docs and citations collected")
                output = get_rag(rephrased_query, retriever, relevant_docs)
                # st.write("op generated.")
                st.info(output)
                if user_input:
                    # response = retrieve_and_generate(query)  # Replace with the actual function call to your RAG implementation
                    st.session_state.query_history.append((user_input, output))
                                # Display citations
                                    # Debugging and checks
                # st.write(f"Type of content: {type(content)}")
                # st.write(f"Type of source: {type(source)}")
                # st.write(f"Type of page: {type(page)}")
                for i, (doc_content, doc_source, doc_page) in enumerate(zip(content, source, page)):
                    with st.expander(f"Citation {i + 1}"):
                        st.write(f"**Content:** {doc_content}")
                        st.write(f"**Source:** {doc_source}")
                        st.write(f"**Page:** {doc_page}")
            
            
            # to show the query history
            with st.sidebar:
                if st.session_state.query_history:
                    st.write("## Query History")
                # messages.chat_message("user").write(prompt)
                # messages.chat_message("assistant").write(f"Echo: {prompt}")
                    for i, (q, r) in enumerate(st.session_state.query_history):
                        st.write(f"**Q{i+1}:** {q}")
                        st.write(f"**A{i+1}:** {r}")
                        st.write("---")
        # return rag_result
    except Exception as e:
        # return(e)
        st.write("Response:", e)

if __name__ == "__main__":
    main()