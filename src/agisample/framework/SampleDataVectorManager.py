import os
from pathlib import Path
from typing import Type

from dotenv import load_dotenv, find_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

_ = load_dotenv(find_dotenv())


class SampleDataVectorManager:
    BASE_DIR = Path(__file__).resolve().parents[3]
    INDEX_NAME = "sample_index"
    DB_LOCAL_FILE_NAME = "sample_db"

    def __init__(self):
        self.__embedding = OpenAIEmbeddings()

    def save(self, documents=Type[list[Document]]):
        faiss_client = FAISS.from_documents(documents, self.__embedding)
        faiss_client.save_local(
            os.path.join(SampleDataVectorManager.BASE_DIR, "data", SampleDataVectorManager.DB_LOCAL_FILE_NAME),
            SampleDataVectorManager.INDEX_NAME)

    def retriever(self):
        db_local_file_path = os.path.join(SampleDataVectorManager.BASE_DIR, "data",
                                          SampleDataVectorManager.DB_LOCAL_FILE_NAME)
        faiss_client = FAISS.load_local(db_local_file_path, self.__embedding, SampleDataVectorManager.INDEX_NAME,
                                        allow_dangerous_deserialization=True)
        return faiss_client.as_retriever()

    def search(self, query):
        docs = self.retriever().get_relevant_documents(query)
        return docs[0].page_content
