# importing all the libraries

import streamlit as st
# from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.load import loads, dumps
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

# using either of the functions to flatten the docs. 
def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

def pretty_print_docs(docs):
    return (f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))


# llm initialization
def get_llm():
    return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# define retriever
def get_retriever(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(docs)
    # print("openai embeddings")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    # print("vector store loaded")
    return vectorstore.as_retriever(search_kwargs={"k": 3})
    

# query rephrase
def get_query_rephrased(query):
    # Rephrasing the query to make it more detailed and clear
    qa_rephrase_template = (
        "Your task is to rephrase user queries to make them more detailed, clear, and specific. "
        "For example, if the user asks 'Tell me about the moon,' a better version might be 'Can you provide a detailed overview of the moon, "
        "including its physical characteristics, phases, and its significance to Earth? Do not use first person reference. "
        "Question might look like this- {query}"
    )
    prompt = PromptTemplate.from_template(qa_rephrase_template)
    # print("qa prompt created")
    llm = get_llm()

    # Chain to rephrase the query
    chain_for_rephrasal = prompt | llm
    # Rephrasing the query
    rephrasal_output = chain_for_rephrasal.invoke({"query": query})
    # print("query rephrased:", rephrasal_output.content)
    return rephrasal_output.content


# citations with source, page number and content 
def get_citations(query,retriever):
    rephrased_query = get_query_rephrased(query)
    docs = retriever.get_relevant_documents(rephrased_query)
    contents = []
    sources = []
    pages = []

    if docs:
        for document in docs:
            contents.append(document.page_content)
            sources.append(document.metadata.get('source', 'Unknown'))
            pages.append(document.metadata.get('page', 'Unknown'))
    else:
        contents.append("")
        sources.append("Unknown")
        pages.append("Unknown")

    return docs,contents,sources,pages


# main rag function
def get_rag(query,retriever, relevant_docs):
    # rephrased_query = get_query_rephrased(query)
    qa_rephrase_template = (
        "Your task is to rephrase user queries to make them more detailed, clear, and specific. "
        "For example, if the user asks 'Tell me about the moon,' a better version might be 'Can you provide a detailed overview of the moon, "
        "including its physical characteristics, phases, and its significance to Earth? Do not use first person reference. "
        "Question might look like this- {query}"
    )
    prompt = PromptTemplate.from_template(qa_rephrase_template)
    # print("qa prompt created")

    llm = get_llm()

    ans_prompt = (
        """
        You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved {context} to answer the
        {query}. Do not use the first person point of view while answering the question.
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.
        """
    )
    qa_system_prompt = PromptTemplate(input_variables=["context", "query"], template=ans_prompt)
    # print("qa prompt created")

    # Define the RAG chain
    rag_chain = (
        {"context":retriever|pretty_print_docs, "query": RunnablePassthrough()}
        | qa_system_prompt
        | llm
        | StrOutputParser()
    )

    # print("rag chain created")

    # Invoke the chain with the correct input type
    # input_data = {"context": doc, "question": modified_query}
    input_data = (query)

    output = rag_chain.invoke(input_data)
    # print("output generated")

    return output