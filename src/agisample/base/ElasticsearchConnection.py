import os
from typing import Optional, List

import elasticsearch

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())


class ElasticsearchConnection():
    def __init__(self, hosts: Optional[List[str]] = None, user=None, password=None):
        if hosts is not None:
            self.__hosts = hosts
        else:
            self.__hosts = os.getenv('ELASTICSEARCH_URL').split(',')
        if user is not None:
            self.__user = user
        else:
            self.__user = os.getenv('ELASTICSEARCH_NAME')
        if password is not None:
            self.__password = password
        else:
            self.__password = os.getenv('ELASTICSEARCH_PASSWORD')
        self.__es_connection = elasticsearch.Elasticsearch(
            hosts=self.__hosts,
            max_retries=10
        )

    def get_user(self):
        return self.__user

    def get_password(self):
        return self.__password

    def get_connection(self):
        return self.__es_connection


if __name__ == '__main__':
    print(ElasticsearchConnection().getConnection())
