from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_environment = os.environ.get("PINECONE_ENVIRONMENT")

pc=Pinecone(api_key=pinecone_api_key)

index_name = "resume-index"

def pinecone_init():
    if index_name not in pc.list_indexes().names():
        index_model = pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region=pinecone_environment,
        embed={
            "model":"multilingual-e5-large",
            "field_map":{"text": "chunk_text"}
        })
    return index_model

