from langchain_openai import OpenAIEmbeddings
from scipy.spatial.distance import cosine


openai_embedding = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key="",
    base_url=""
)

text0 = "None-None#Reference Number-4000058247#None-12.11.2024#Payment amt-158,765,479.00"
text1 = "None-None#Invoice No-RV240207021#Date-12.11.2024#Total-158,765,479.00"
text2 = "None-None#Reference Number-RV240207021#None-12.11.2024#Total-158,765,479.00"
text3 = "None-None#Reference Number-5111156235#Date-12.11.2024#Payment amt-158,765,479.00"
text4 = "None-None#Reference Number-5111156235#None-12.11.2024#Payment amt-158,765,479.00"

embedding0 = openai_embedding.embed_query(text0)
embedding1 = openai_embedding.embed_query(text1)
embedding2 = openai_embedding.embed_query(text2)
embedding3 = openai_embedding.embed_query(text3)
embedding4 = openai_embedding.embed_query(text4)

cosine_distance_0_1 = cosine(embedding0, embedding1)
cosine_distance_0_2 = cosine(embedding0, embedding2)
cosine_distance_0_3 = cosine(embedding0, embedding3)
cosine_distance_0_4 = cosine(embedding0, embedding4)


print("Cosine Distance 0-1:", cosine_distance_0_1)
print("Cosine Distance 0-2:", cosine_distance_0_2)
print("Cosine Distance 0-3:", cosine_distance_0_3)
print("Cosine Distance 0-4:", cosine_distance_0_4)

