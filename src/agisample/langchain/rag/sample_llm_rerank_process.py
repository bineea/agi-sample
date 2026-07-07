from langchain_classic.retrievers.document_compressors.listwise_rerank import (
    LLMListwiseRerank,
)
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI


documents = [
    Document("Invoice No"),
    Document("Reference Number"),
    Document("Reference"),
    Document("Number"),
    Document("PO NO"),
    Document("GRN No"),
]


def rerank_documents(query: str, top_n: int = 3) -> list[Document]:
    reranker = LLMListwiseRerank.from_llm(
        llm=ChatOpenAI(model="gpt-4o-mini"), top_n=top_n
    )
    return reranker.compress_documents(documents, query)


def main() -> None:
    compressed_docs = rerank_documents("the billing reference of the related order")
    print(compressed_docs)
    assert len(compressed_docs) == 3
    assert all(doc.page_content for doc in compressed_docs)


if __name__ == "__main__":
    main()

