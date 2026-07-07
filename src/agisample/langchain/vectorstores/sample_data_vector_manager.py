import os
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


class SampleDataVectorManager:
    BASE_DIR = Path(__file__).resolve().parents[3]
    INDEX_NAME = "sample_index"
    DB_LOCAL_FILE_NAME = "sample_db"

    def __init__(self):
        load_dotenv(find_dotenv())
        self.__embedding = OpenAIEmbeddings()

    def save(self, documents: list[Document]) -> None:
        faiss_client = FAISS.from_documents(documents, self.__embedding)
        faiss_client.save_local(
            os.path.join(SampleDataVectorManager.BASE_DIR, "data", SampleDataVectorManager.DB_LOCAL_FILE_NAME),
            SampleDataVectorManager.INDEX_NAME,
        )

    def retriever(self):
        db_local_file_path = os.path.join(
            SampleDataVectorManager.BASE_DIR,
            "data",
            SampleDataVectorManager.DB_LOCAL_FILE_NAME,
        )
        faiss_client = FAISS.load_local(
            db_local_file_path,
            self.__embedding,
            SampleDataVectorManager.INDEX_NAME,
        )
        return faiss_client.as_retriever()

    def search(self, query):
        docs = self.retriever().invoke(query)
        return docs[0].page_content

