import os
import logging
from dotenv import load_dotenv
from pymilvus import connections, Collection
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Embeddings, ModelInference
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes, ModelTypes

# ─── Setup ─────────────────────────────────────────────────────────────

# Load .env values
load_dotenv(override=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

COS_API_KEY = os.getenv("COS_API_KEY")
WATSON_STUDIO_PROJECT_ID = os.getenv("WATSON_STUDIO_PROJECT_ID")
MILVUS_HOST = os.getenv("MILVUS_GRPC_HOST")
MILVUS_PORT = int(os.getenv("MILVUS_GRPC_PORT"))
MILVUS_CERT_PATH = os.getenv("MILVUS_CERT_PATH")
# ─── Watsonx Credentials ────────────────────────────────────────────────

logging.info("Initializing Watsonx credentials")
client = Credentials(url="https://us-south.ml.cloud.ibm.com", api_key=COS_API_KEY)

logging.info("Loading embedding model")
embedding = Embeddings(
    model_id=EmbeddingTypes.IBM_SLATE_30M_ENG,
    project_id=WATSON_STUDIO_PROJECT_ID,
    credentials=client
)

logging.info("Loading LLM (ModelInference)")
llm = ModelInference(
    model_id=ModelTypes.GRANITE_13B_INSTRUCT_V2,   # ibm/granite-13b-instruct-v2
    project_id=WATSON_STUDIO_PROJECT_ID,
    credentials=client
)

# ─── Connect to Milvus ──────────────────────────────────────────────────

connections.connect(
    alias="default",
    host=MILVUS_HOST,
    port=MILVUS_PORT,
    user="ibmlhapikey",
    password=COS_API_KEY,
    secure=True,
    server_ca=MILVUS_CERT_PATH
)

# ─── Generation Function ────────────────────────────────────────────────

def generate_answer(query: str, collection_name: str = "carbon_embeddings", top_k: int = 5) -> str:
    logging.info(f"Generating answer for query: {query}")

    try:
        # Step 1: Embed the query
        query_vector = embedding.embed_documents([query])[0]
        logging.debug("Query embedding generated successfully")

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
        logging.debug(f"Milvus search completed, found {len(results)} results")

        # Step 3: Gather context
        context_chunks = []
        for hit in results[0]:
            chunk = hit.entity.get("chunk_text")
            file_name = hit.entity.get("file_name")
            if chunk and file_name:
                context_chunks.append(f"From {file_name}:\n{chunk.strip()}")
        
        if not context_chunks:
            logging.warning("No relevant context found in the search results")
            return "I cannot find information about that in the available data."

        context = "\n\n".join(context_chunks)
        logging.info(f"Retrieved {len(context_chunks)} relevant chunks from Milvus")

        # Step 4: Construct prompt
        prompt = f"""Use the context below to answer the user's question as clearly and accurately as possible.
If the answer cannot be found in the context, say "I cannot find information about that in the available data."

Context:
{context}

Question:
{query}

Answer:"""

        # Step 5: Generate response from Watsonx LLM
        response = llm.generate(prompt=prompt)
        answer = response['results'][0]['generated_text']
        return answer

    except Exception as e:
        logging.error(f"Error in generate_answer: {str(e)}", exc_info=True)
        return f"An error occurred while processing your question: {str(e)}"

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
