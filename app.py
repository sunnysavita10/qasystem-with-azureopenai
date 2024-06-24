from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from langchain.vectorstores import FAISS
from qasystem.ingestion import data_ingestion,get_vector_store

import json
import os 
import sys
import streamlit as st

from langchain_openai import OpenAIEmbeddings
from langchain_openai import AzureOpenAI

Embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
from qasystem.retrievalandgeneration import get_llm,get_response_llm

def main():
    st.set_page_config("QA with Doc")
    st.header("QA with Doc using langchain and AzureOpenAI")
    
    user_question=st.text_input("Ask a question from the pdf files")
    
    with st.sidebar:
        st.title("update or create the vector store")
        if st.button("vectors update"):
            with st.spinner("processing..."):
                docs=data_ingestion()
                get_vector_store(docs)
                st.success("done")
                
        if st.button("azure model"):
            with st.spinner("processing..."):
                faiss_index=FAISS.load_local("faiss_index",Embedding,allow_dangerous_deserialization=True)
                llm=get_llm()
                
                st.write(get_response_llm(llm,faiss_index,user_question))
                st.success("Done")
                
if __name__=="__main__":
    #this is my main method
    main()