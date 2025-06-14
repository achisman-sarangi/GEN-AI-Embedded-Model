from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

OpenAIEmbeddings = OpenAIEmbeddings(model= '', dimensions = 30)

documents = [
    "Narendra Modi is the current prime minister of india",
    "The capital of india is New Delhi",
    "The capital Of The USA is Washington DC",
    "The capital of japan is Tokyo"
]

result = OpenAIEmbeddings.embed_documents(documents)

print(str(result))