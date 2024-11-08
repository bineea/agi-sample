from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI


# 通过LLM计算差值不准确，需要通过具体方法计算差值
# 最好将单条处理和组合处理拆分为多个prompt
class MatchAgentProcess:

    PROMPT = """
    你是专门用于匹配payment数据和ar数据的AI助手。
    
    请按照以下步骤处理数据：
        1. 解析{payment}数据：读取并解析JSON结构的payment数据。
        2. 解析{ar}数据集合：读取并解析JSON结构的ar数据集合。
        3. 比较与匹配：
            a. 遍历ar数据集合中的每一条数据。
            b. amount字段比较：比较payment数据和ar数据的amount字段。根据amount字段数值差值，赋予0%到55%的匹配度，并且数值差值越小匹配度越高，若数值差值小于1，则直接赋予60%的匹配度。
            c. reference字段比较：比较payment数据和ar数据的reference字段。若两者语义一致或存在包含关系（即一个字符串是另一个字符串的子串），则赋予0%~20%的匹配度。
        4. 根据上述规则，计算每条ar数据的总匹配度，
        5. 确定最佳匹配：
            a. 如果存在匹配度达到或超过60%的ar数据，则返回匹配度最高的ar数据，
            b. 如果每条ar数据的总匹配度均低于60%，则查找多条ar数据的amount字段数值的总和与payment数据的amount字段数值差异小于1的ar数据组合，并直接返回ar数据组合，
    
    注意事项：
        - 避免编造或猜测任何信息。
        - 匹配度计算应基于给定的规则，不得随意变动。
        - 只返回最终结果，不需要返回中间过程。

    示例：
    示例1：
    payment数据：{{"payment_doc_no": "5900000027", "amount": 1886.25, "reference": "6391261858"}}
    ar数据集合：[{{"ar_doc_no": "9660007437", "amount": 1886.25, "reference": "6391261858"}}]
    助手：比较payment数据和ar数据，匹配度超过60%，匹配度成功的ar数据为{{"ar_doc_no": "9660007437", "amount": 1886.25, "reference": "6391261858"}}。
    
    示例2：
    payment数据：{{"payment_doc_no": "5900000027", "amount": 1886.25, "reference": "6391261858"}}
    ar数据集合：[{{"ar_doc_no": "9660007437", "amount": 1886.25, "reference": "146209"}}]
    助手：比较payment数据和ar数据，匹配度达到60%，匹配度成功的ar数据为{{"ar_doc_no": "9660007437", "amount": 1886.25, "reference": "146209"}}。
    
    示例3：
    payment数据：{{"payment_doc_no": "5900000027", "amount": 1886.25, "reference": "6391261858"}}
    ar数据集合：[{{"ar_doc_no": "9660007437", "amount": 1886, "reference": "146209"}},{{"ar_doc_no": "9660007438", "amount": 1886.05, "reference": "146209"}}]
    助手：比较payment数据和ar数据，最佳匹配为第二条ar数据，匹配度成功的ar数据为{{"ar_doc_no": "9660007438", "amount": 1886.05, "reference": "146209"}}。
    
    示例4：
    payment数据：{{"payment_doc_no": "5900000027", "amount": 1886.25, "reference": "6391261858"}}
    ar数据集合：[{{"ar_doc_no": "9660007437", "amount": 886.25, "reference": "146209"}}, {{"ar_doc_no": "9660007438", "amount": 1000, "reference": "146209"}}]
    助手：比较payment数据和ar数据，每条ar数据匹配度均低于60%，组合ar数据匹配度成功，结果为[{{"ar_doc_no": "9660007437", "amount": 86.25, "reference": "146209"}}, {{"ar_doc_no": "9660007438", "amount": 1000, "reference": "146209"}}]。
    
    示例5：
    payment数据：{{"payment_doc_no": "5900000027", "amount": 1886.25, "reference": "6391261858"}}
    ar数据集合：[{{"ar_doc_no": "9660007437", "amount": 986.25, "reference": "146209"}},{{"ar_doc_no": "9660007438", "amount": 900, "reference": "146209"}}, {{"ar_doc_no": "9660007439", "amount": 1000, "reference": "146209"}}]
    助手：比较payment数据和ar数据，每条ar数据匹配度均低于60%，组合ar数据匹配度成功，结果为[{{"ar_doc_no": "9660007437", "amount": 986.25, "reference": "146209"}}, {{"ar_doc_no": "9660007438", "amount": 900, "reference": "146209"}}]。
    
    示例6：
    payment数据：{{"payment_doc_no": "5900000027", "amount": 1886.25, "reference": "6391261858"}}
    ar数据集合：[{{"ar_doc_no": "9660007437", "amount": 86.25, "reference": "146209"}}]
    助手：比较payment数据和ar数据，匹配度未达到60%，没有找到匹配的ar数据。
    
    """

    PROMPT_TEMPLATE = ChatPromptTemplate.from_template(PROMPT)

    def rag(self):
        payment = {"payment_doc_no": "5900000027", "amount": 2580.99, "reference": "6391261858"}
        ar = [
                {"ar_doc_no": "9660007437", "amount": 925432.12, "reference": "146209"},
                {"ar_doc_no": "9660007438", "amount": 1818.39, "reference": "146209"},
                {"ar_doc_no": "9660007439", "amount": 21502.80, "reference": "146209"},
                {"ar_doc_no": "9660007440", "amount": 2211.00, "reference": "146209"},
                {"ar_doc_no": "9660007441", "amount": 2924.24, "reference": "146209"},
                {"ar_doc_no": "9660007442", "amount": 303911.84, "reference": "146209"},
                {"ar_doc_no": "9660007443", "amount": 536166.17, "reference": "146209"},
                {"ar_doc_no": "9660007444", "amount": -4092.00, "reference": "146209"},
                {"ar_doc_no": "9660007445", "amount": 318.79, "reference": "146209"},
                {"ar_doc_no": "9660007446", "amount": 58089.90, "reference": "146209"},
                {"ar_doc_no": "9660007446", "amount": 2580.90, "reference": "146209"},
        ]
        rag_chain = (
                RunnablePassthrough()
                | MatchAgentProcess.PROMPT_TEMPLATE
                | ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key="", base_url="")
                | StrOutputParser()
        )
        return rag_chain.invoke({"payment": payment, "ar": ar})


if __name__ == '__main__':
    print(MatchAgentProcess().rag())