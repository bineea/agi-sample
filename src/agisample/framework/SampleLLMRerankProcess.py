from langchain.retrievers.document_compressors.listwise_rerank import (
    LLMListwiseRerank,
)
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

# documents = [
#     Document("Sally is my friend from school"),
#     Document("Steve is my friend from home"),
#     Document("I didn't always like yogurt"),
#     Document("I wonder why it's called football"),
#     Document("Where's waldo"),
# ]


documents = [
    Document("Invoice No"),
    Document("Reference Number"),
    Document("Reference"),
    Document("Number"),
    Document("PO NO"),
    Document("GRN No")
]


reranker = LLMListwiseRerank.from_llm(
    llm=ChatOpenAI(model="gpt-4o-mini"), top_n=3
)
# compressed_docs = reranker.compress_documents(documents, "Who is steve")
compressed_docs = reranker.compress_documents(documents, "the billing reference of the related order")
print(compressed_docs)
assert len(compressed_docs) == 3
assert "Steve" in compressed_docs[0].page_content