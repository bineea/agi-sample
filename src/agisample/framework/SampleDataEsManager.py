from typing import List

from dotenv import load_dotenv, find_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import ElasticsearchStore
from langchain_core.documents import Document

from agisample.base.ElasticsearchConnection import ElasticsearchConnection

_ = load_dotenv(find_dotenv())


class SampleDataEsManager:
    INDEX_NAME = "sample_index"

    def __init__(self, es_connection: ElasticsearchConnection = ElasticsearchConnection()):
        self.__embedding = OpenAIEmbeddings()
        self.__esStore = ElasticsearchStore(
            index_name=SampleDataEsManager.INDEX_NAME,
            es_connection=es_connection.get_connection(),
            es_user=es_connection.get_user(),
            es_password=es_connection.get_password(),
            embedding=self.__embedding
        )

    def save(self, documents=List[Document]):
        print("save")

    def search(self, query):
        print("search")
