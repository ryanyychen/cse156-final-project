#### Imports ####
import os
import torch
import time
import gradio as gr
from dotenv import load_dotenv

# Embeddings #
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5ForConditionalGeneration

# Database #
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams

# Gemini
import google.generativeai as genai

from embeddings import vectorize
from retrieval import hybrid_query, sparse_query, dense_query, bm25_query

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print("Using " + DEVICE)

load_dotenv()

def start_database(recreate, sparse_vector_size, dense_vector_size, collection_name):
    try:
        client = QdrantClient(
            url="https://07de6745-b3ea-4156-9daf-a4cbbb339b92.us-east4-0.gcp.cloud.qdrant.io:6333", 
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        print("Connected to Qdrant database")
    except Exception as e:
        print(e)
        return None

    print(f"Using sparse vector size: {sparse_vector_size}, dense vector size: {dense_vector_size}")

    # If recreate=True, recreate the collection
    if recreate:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config={
                "dense_vector": VectorParams(size=dense_vector_size, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse_vector": SparseVectorParams(),
            },
        )

    # Create collection if it doesn't exist
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense_vector": VectorParams(size=dense_vector_size, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse_vector": SparseVectorParams(),
            },
        )
        
    print("Existing collections:")
    print(client.get_collections())

    return client

# Initialize tokenizers and embedding models
SPARSE_MODEL_NAME = "naver/splade_v2_distil"
sparse_tokenizer = AutoTokenizer.from_pretrained(
    SPARSE_MODEL_NAME,
    device=DEVICE,
)
sparse_model = AutoModel.from_pretrained(SPARSE_MODEL_NAME).to(DEVICE)
SPARSE_VECTOR_SIZE = 50000  # Sparse embeddings can get very large
print("Sparse model " + SPARSE_MODEL_NAME + " initialized")

# Dense embedder
DENSE_MODEL_NAME = "intfloat/e5-base"
dense_model = SentenceTransformer(
    DENSE_MODEL_NAME,
    device=DEVICE,
)
DENSE_VECTOR_SIZE = dense_model.get_sentence_embedding_dimension()
print("Dense model " + DENSE_MODEL_NAME + " initialized")

# Database
COLLECTION_NAME = "hybrid_retrieval"

# LLM
tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(DEVICE)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-1.5-pro-latest')


# Initialize the database connection
db = start_database(recreate=False, sparse_vector_size=SPARSE_VECTOR_SIZE, dense_vector_size=DENSE_VECTOR_SIZE, collection_name=COLLECTION_NAME)


def get_model_output(client, query, retrieval_type="hybrid", top_k=5, model="gemini"):
    """
    Queries the Qdrant vector database and generates an answer using a T5 model.

    Args:
        client: QdrantClient instance.
        query (str): The input question.
        retrieval_type (str): Type of vector retrieval ('sparse', 'dense', 'hybrid').
        top_k (int): Number of top retrieved documents.

    Returns:
        str: Generated answer.
    """
    start_time = time.time()

    # Generate embeddings
    query_sparse, query_dense = vectorize(query, sparse_tokenizer, sparse_model, SPARSE_VECTOR_SIZE, dense_model, prefix="query: ")
    query_dense = query_dense.cpu().numpy()

    if retrieval_type == "sparse":
        results = sparse_query(client, query_sparse, top_k, COLLECTION_NAME)
        context_list = [point.payload["text"] for point in results.points]
    elif retrieval_type == "dense":
        results = dense_query(client, query_dense, top_k, COLLECTION_NAME)
        context_list = [point.payload["text"] for point in results.points]
    elif retrieval_type == "bm25":
        results = bm25_query(client, query, COLLECTION_NAME, top_k)
        context_list = results
    else:  # Hybrid retrieval
        results = hybrid_query(client, query, top_k, sparse_tokenizer, sparse_model, SPARSE_VECTOR_SIZE, dense_model, COLLECTION_NAME)
        context_list = [result['text'] for result in results]

    context = " ".join(context_list)
    end_time = time.time()
    print(f"Context retrieved in {end_time - start_time:.2f} seconds")
    input_text = f"According to this context, and only this context: {context} Answer this: {query} "

    start_time = time.time()

    # Load model and tokenizer
    if model == "t5":
        with torch.no_grad():
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(DEVICE)
            output = t5_model.generate(input_ids, max_length=500)
        answer = tokenizer.decode(output[0], skip_special_tokens=True)

    elif model == "gemini":
        response = gemini_model.generate_content(input_text)
        answer = response.text
    end_time = time.time()
    print(f"Answer generated in {end_time - start_time:.2f} seconds")
    return answer

def respond(message, chat_history):
    retrieval_method = retrieval_type.value
    k = top_k.value
    model = model_type.value
    return get_model_output(
        client=db,
        query=message,
        retrieval_type=retrieval_method,
        top_k=k,
        model=model,
    )


# Create a Gradio interface
with gr.Blocks() as demo:
    # Add description at the top
    gr.Markdown("""
    # CSE 156 Personal TA
    This is an AI assistant that can answer questions about class materials. 
    The system retrieves relevant content from lecture slides and answers based on that information.
    """)
    
    # Create dropdown for retrieval type and model
    with gr.Row():
        retrieval_type = gr.Dropdown(
            choices=["hybrid", "sparse", "dense", "bm25"], 
            value="hybrid", 
            label="Retrieval Method"
        )
        model_type = gr.Dropdown(
            choices=["gemini", "t5"], 
            value="gemini", 
            label="Model"
        )
        top_k = gr.Slider(
            minimum=1, 
            maximum=10, 
            value=5, 
            step=1, 
            label="Number of documents to retrieve"
        )
    
    # Chat interface using gr.ChatInterface
    chat = gr.ChatInterface(
        respond,
        chatbot=gr.Chatbot(placeholder="What would you like to know?"),
        textbox=gr.Textbox(placeholder="Ask a question about the course...", show_label=False)
    )

    clear = gr.Button("Clear")
    clear.click(lambda: None, outputs=chat)
# Launch the app
if __name__ == "__main__":
    demo.launch(share=False)