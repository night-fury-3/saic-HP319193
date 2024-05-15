from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

source_path = "car.txt"
loader = TextLoader(source_path, encoding="utf-8")

data = loader.load()

print(data)

text_splitter = CharacterTextSplitter(separator = "##", chunk_size=300, chunk_overlap=0, length_function = len)

texts = text_splitter.split_documents(data)

db = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")



