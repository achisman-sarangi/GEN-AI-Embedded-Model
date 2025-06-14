from langchain_huggingface import HuggingFaceEmbeddings

embedding  = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

text = "Bhubaneswar is the capital of odisha"

result = embedding.embed_query(text)

print(str(result))