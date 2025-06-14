from langchain_huggingface import HuggingFaceEmbeddings

embedding  = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Narendra Modi is the current prime minister of india",
    "The capital of india is New Delhi",
    "The capital Of The USA is Washington DC",
    "The capital of japan is Tokyo"
]

result = embedding.embed_documents(documents)

print(str(result))