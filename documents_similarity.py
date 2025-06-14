from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model = 'text-embedding-3-large', dimensions = 500)

documents = [
    "Narendra modi is the current prime minister of india.And he is the leader of the BJP party",
    "Virat kohli is the best ODI player in the world now.And his batting skills are really impressive",
    "The capital of oissha is bhubaneswar and also one of the smartest city in india",
    "Salman khan is the bollywood actor and he is one of the most popular actor in india",
    "Christiano ronaldo is the best football player in the world and he is the captain of the portuagal national team"
]

query = "Tell me about  salman khan"

documents_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

similarity_score = cosine_similarity([query_embedding],documents_embeddings)[0]
index,score = sorted(list(enumerate(similarity_score)),key= lambda x:x[1])[-1]

print(query)
print(documents[index])
print("similariy score is:", score)