from langchain_openai import OpenAIEmbeddings
from scipy.spatial.distance import cosine, euclidean

# 欧氏距离（Euclidean Distance）是最易于理解的距离度量方法之一。我们可以通过将两点之间的直线距离计算出来来计算两点之间的距离。
# 余弦相似度（Cosine Similarity）是通过计算两个向量之间的夹角余弦值来评估它们之间的相似性。余弦相似度的取值范围在-1到1之间，1表示两个向量方向完全相同，0表示两个向量是独立的，-1表示两个向量方向完全相反。
# 欧氏距离 (Euclidean Distance) 取值范围：[0, +∞)
# 余弦距离 (Cosine Distance) 取值范围：[0, 2]


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


text10 = "{\"sourceField\": \"\", \"value\": \"\"}"
text11 = "{\"sourceField\": \"Reference Number\", \"value\": \"4000058247\"}"
text12 = "{\"sourceField\": \"\", \"value\": \"12.11.2024\"}"
text13 = "{\"sourceField\": \"Payment amt\", \"value\": \"158,765,479.00\"}"
text14 = "{\"sourceField\": \"\", \"value\": \"\"}"
text15 = "{\"sourceField\": \"\", \"value\": \"\"}"
text16 = "{\"sourceField\": \"Invoice No\", \"value\": \"RV240207021238\"}"
text17 = "{\"sourceField\": \"Gross Amount\", \"value\": \"2,373,365.86\"}"
text18 = "{\"sourceField\": \"Less TDS Amount\", \"value\": \"2,011.00\"}"
text19 = "{\"sourceField\": \"\", \"value\": \"\"}"
text110 = "{\"sourceField\": \"\", \"value\": \"\"}"

embedding10_110 = openai_embedding.embed_documents([text10, text11, text12, text13, text14, text15, text16, text17, text18, text19, text110])
embedding10 = openai_embedding.embed_query(text0)

cosine_distance_10_110_1 = cosine(embedding10_110[0], embedding10)
cosine_distance_10_110_4 = cosine(embedding10_110[4], embedding10)
cosine_distance_10_110_5 = cosine(embedding10_110[5], embedding10)
cosine_distance_10_110_9 = cosine(embedding10_110[9], embedding10)
cosine_distance_10_110_10 = cosine(embedding10_110[10], embedding10)


euclidean_distance_10_110_1 = euclidean(embedding10_110[0], embedding10)
euclidean_distance_10_110_4 = euclidean(embedding10_110[4], embedding10)
euclidean_distance_10_110_5 = euclidean(embedding10_110[5], embedding10)
euclidean_distance_10_110_9 = euclidean(embedding10_110[9], embedding10)
euclidean_distance_10_110_10 = euclidean(embedding10_110[10], embedding10)


print("Cosine Distance 10_110-1:", cosine_distance_10_110_1)
print("Cosine Distance 10_110-4:", cosine_distance_10_110_4)
print("Cosine Distance 10_110-5:", cosine_distance_10_110_5)
print("Cosine Distance 10_110-9:", cosine_distance_10_110_9)
print("Cosine Distance 10_110-10:", cosine_distance_10_110_10)

print("Euclidean Distance 10_110-1:", euclidean_distance_10_110_1)
print("Euclidean Distance 10_110-4:", euclidean_distance_10_110_4)
print("Euclidean Distance 10_110-5:", euclidean_distance_10_110_5)
print("Euclidean Distance 10_110-9:", euclidean_distance_10_110_9)
print("Euclidean Distance 10_110-10:", euclidean_distance_10_110_10)