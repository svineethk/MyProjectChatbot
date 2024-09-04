import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

import time
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt
from openai import RateLimitError

os.environ['OPENAI_API_KEY'] = "api-key"

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_embeddings_safe(texts):
    return openai.Embedding.create(input=texts, model="text-embedding-ada-002")

def build_index(docs):
    try:
        index = VectorStoreIndex.from_documents(docs)
        return index
    except RateLimitError as e:
        print(f"Rate limit error: {e}")
        time.sleep(10)
        return build_index(docs)


docs = SimpleDirectoryReader('./dataset').load_data()
index = build_index(docs)

chat_engine = index.as_chat_engine(
    response_mode="compact"
)

def get_response(msg):
    response = chat_engine.chat(msg)
    return response

user_input = "education"
print(get_response(user_input))
