from typing import List

from dotenv import load_dotenv, find_dotenv
from langchain_core.documents import Document
from langchain_elasticsearch import ElasticsearchStore
from langchain_openai import OpenAIEmbeddings

from agisample.common.elasticsearch_connection import ElasticsearchConnection


class SampleDataEsManager:
    INDEX_NAME = "sample_index"

    def __init__(self, es_connection: ElasticsearchConnection = ElasticsearchConnection()):
        load_dotenv(find_dotenv())
        self.__embedding = OpenAIEmbeddings()
        self.__esStore = ElasticsearchStore(
            index_name=SampleDataEsManager.INDEX_NAME,
            es_connection=es_connection.get_connection(),
            es_user=es_connection.get_user(),
            es_password=es_connection.get_password(),
            embedding=self.__embedding,
        )

    def save(self, documents: List[Document]):
        print("save")

    def search(self, query):
        print("search")

