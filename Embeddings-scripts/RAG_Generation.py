import os
import logging
from dotenv import load_dotenv
from pymilvus import connections, Collection
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Embeddings, Model
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes, ModelTypes

# ─── Setup ─────────────────────────────────────────────────────────────

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

COS_API_KEY = os.getenv("COS_API_KEY")
WATSON_STUDIO_PROJECT_ID = os.getenv("WATSON_STUDIO_PROJECT_ID")

MILVUS_HOST = os.getenv("MILVUS_GRPC_HOST")
MILVUS_PORT = int(os.getenv("MILVUS_GRPC_PORT"))

# ─── Watsonx Credentials ────────────────────────────────────────────────
logging.info(f"Embedding Model loading")
client = Credentials(url="https://us-south.ml.cloud.ibm.com", api_key=COS_API_KEY)
embedding = Embeddings(
    model_id=EmbeddingTypes.IBM_SLATE_30M_ENG,
    project_id=WATSON_STUDIO_PROJECT_ID,
    credentials=client
)
logging.info(f"Embedding Model loaded")

logging.info(f"LLM Model loading")
llm = Model(
    model_id=ModelTypes.GRANITE_13B_CHAT_V2,
    project_id=WATSON_STUDIO_PROJECT_ID,
    credentials=client
)
logging.info(f"LLM Model loaded")

# ─── Connect to Milvus ──────────────────────────────────────────────────

connections.connect(
    alias="default",
    host=MILVUS_HOST,
    port=MILVUS_PORT,
    user="ibmlhapikey",
    password=COS_API_KEY,
    secure=True
)

# ─── Generation Function ────────────────────────────────────────────────

def generate_answer(query: str, collection_name: str = "carbon_embeddings", top_k: int = 5) -> str:
    logging.info(f"Generating answer for query: {query}")

    # Step 1: Embed the query
    query_vector = embedding_model.embed_documents([query])[0]

    # Step 2: Search Milvus
    collection = Collection(name=collection_name)
    collection.load()
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["chunk_text", "file_name"]
    )

    # Step 3: Gather context
    context_chunks = []
    for hit in results[0]:
        chunk = hit.entity.get("chunk_text", "")
        context_chunks.append(chunk.strip())
    context = "\n".join(context_chunks)

    logging.info(f"Retrieved {len(context_chunks)} relevant chunks from Milvus.")

    # Step 4: Construct prompt
    prompt = f"""Use the context below to answer the user's question as clearly and accurately as possible.

Context:
{context}

Question:
{query}

Answer:"""

    # Step 5: Generate response from Watsonx LLM
    response = llm.generate(prompt=prompt)
    answer = response['results'][0]['generated_text']
    return answer

# ─── Main Loop ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("💬 CarbonSense | Ask your sustainability question")
    while True:
        try:
            query = input("\n❓ Question (or type 'exit'): ").strip()
            if query.lower() == "exit":
                break

            answer = generate_answer(query)
            print("\n🧠 Answer:\n", answer)

        except Exception as e:
            logging.error(f"Error during generation: {e}")
