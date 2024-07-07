import os
from pathlib import Path

from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agisample.framework.SampleDataVectorManager import SampleDataVectorManager


class SampleRagProcess():

    BASE_DIR = Path(__file__).resolve().parents[3]

    def init_data(self):
        pdf_loader = PyMuPDFLoader(os.path.join(SampleDataVectorManager.BASE_DIR, "docs", "llama2.pdf"))
        pages = pdf_loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
        )
        split_docs = text_splitter.create_documents(
            [page.page_content for page in pages[:10]]
        )
        SampleDataVectorManager().save(split_docs)

    # PROMPT = """Answer the question: {question} based only on the following context: {context}. The answer must not contain external information beyond the context provided. If unable to answer the question based on the provided context, reply directly: unable to answer the question based on known information"""
    #
    # PROMPT_TEMPLATE = ChatPromptTemplate.from_template(TEMPLATE)

    PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
        (
            "system",
            "Answer the human question based only on the following context: \n{context}.\n The answer must not contain external information beyond the context provided. If unable to answer the question based on the provided context, reply directly: According to the content you provided, there was no direct mention of {question}.",
        ),
        (
            "human",
            "{question}"
        )
    ])

    def rag(self, question):
        retriever = SampleDataVectorManager().retriever()
        rag_chain = (
                {"question": RunnablePassthrough(), "context": retriever}
                | SampleRagProcess.PROMPT_TEMPLATE
                | ChatOpenAI(model="gpt-4o", temperature=0)
                | StrOutputParser()
        )
        return rag_chain.invoke(question)


if __name__ == '__main__':
    # print(SampleRagProcess.PROMPT)
    print("-----------")
    print(SampleDataVectorManager().search("llama2有多少参数？"))
    print("-----------")
    print(SampleRagProcess().rag("llama2有多少参数？"))
    print("-----------")
    print(SampleRagProcess().rag("1加1等于多少？"))
    print("-----------")
