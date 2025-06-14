from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

OpenAIEmbeddings = OpenAIEmbeddings(model= 'text-embedding-3-large', dimensions = 500)

result = OpenAIEmbeddings.embed_query(" Narendra Modi is the current prime minister of india")

print(str(result))