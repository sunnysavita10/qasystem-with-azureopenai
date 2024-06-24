
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from openai import AzureOpenAI
import json
import os
import sys
from langchain_openai import OpenAIEmbeddings
from langchain_openai import AzureOpenAI

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY

Embedding = OpenAIEmbeddings(api_key= OPENAI_API_KEY)

def data_ingestion():
    loader=PyPDFDirectoryLoader("C:\\Complete_Content\\GENERATIVEAI\\GenerativeAIProjects\\qasystemwithazure\\data")
    documents=loader.load()
    
    
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_splitter.split_documents(documents)
    
    docs=text_splitter.split_documents(documents)
    
    return docs

def get_vector_store(docs):
    vector_store_faiss=FAISS.from_documents(docs,Embedding)
    vector_store_faiss.save_local("faiss_index")
    return vector_store_faiss

if __name__ == '__main__':
    docs=data_ingestion()
    print(docs)
    get_vector_store(docs)
