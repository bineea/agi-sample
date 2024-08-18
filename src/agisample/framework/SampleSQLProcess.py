from dotenv import load_dotenv, find_dotenv
from vanna.chromadb import ChromaDB_VectorStore
from vanna.flask import VannaFlaskApp
from vanna.openai import OpenAI_Chat

_ = load_dotenv(find_dotenv())


class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)


class SampleSQLProcess():
    def __init__(self):
        self.__vanna = MyVanna(config={'base_url': 'https://api.fe8.cn/v1', 'api_key': 'sk-...', 'model': 'gpt-4-...'})
        self.__vanna.connect_to_mysql(host='my-host', dbname='my-db', user='my-user', password='my-password', port=123)

    def prepare(self):
        df_information_schema = self.__vanna.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")
        plan = self.__vanna.get_training_plan_generic(df_information_schema)
        self.__vanna.train(plan=plan)
        self.__vanna.train(ddl="""
        CREATE TABLE user  (
          id bigint(0) NOT NULL AUTO_INCREMENT COMMENT '唯一标识',
          user_name varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '姓名',
          user_age int(0) NULL DEFAULT NULL COMMENT '年龄',
          user_gender int(0) NULL DEFAULT NULL COMMENT '性别',
          user_status varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '状态；VALID为有效，INVALID为无效',
          delete_flag tinyint(1) NOT NULL DEFAULT 0 COMMENT '逻辑删除；1为已删除，0为未删除',
          PRIMARY KEY (id) USING BTREE
        ) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '用户信息' ROW_FORMAT = Dynamic;
        """)
        self.__vanna.train(documentation="性别字段的值为数字，0为男性，1为女性")
        self.__vanna.train(sql="SELECT * FROM user WHERE name = 'John Doe'")

    def get(self):
        print(self.__vanna.get_training_data())

    def ask(self, question):
        self.__vanna.ask(question)

    def run(self):
        app = VannaFlaskApp(self.__vanna)
        app.run()


if __name__ == '__main__':
    print("sample sql process")
    sqlProcess = SampleSQLProcess()
    # sqlProcess.prepare()
    # sqlProcess.get()
    # sqlProcess.ask("根据性别和年龄分析用户信息")
    sqlProcess.run()






# <context>
#   表结构信息如下：
#   {{表结构信息}}
# </context>
# <objective>
#   你是一个高级SQL生成器，能够根据不同的SQL方言生成相应的SQL语句。你需要将用户输入的自然语言转化为SQL，请按照以下步骤操作：
#   1. 请一步步思考并仔细分析用户的自然语言输入，确保充分理解用户的意图。
#   2. 识别目标数据库类型为{{SQL方言}} SQL
#   3. 考虑该数据库类型的特定语法和函数。
#   4. 根据理解的用户意图，设计SQL查询的基本结构。
#   5. 应用数据库特定的语法规则，对基本结构进行调整。
#   6. 优化查询以提高性能（如适用）。
#   7. 生成最终的SQL语句。
#
#   在生成SQL时，请特别注意以下几点：
#   - 使用{{SQL方言}} SQL特有的函数和语法结构 - 考虑该数据库类型的查询优化技巧
#   - 确保生成的SQL语句在语法和逻辑上的正确性
#   如果用户的请求不明确或需要额外信息，请提出澄清性问题。
# </objective>