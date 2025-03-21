#### Imports ####
import os
import torch
import time
import gradio as gr
import shutil
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5ForConditionalGeneration
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams

from embeddings import vectorize
from document_upload import read_file, upsert
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

# Text splitter to chunk texts
# Using semantic chunking for best separation of different information to help retrieval
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_amount=0.2, # Higher = fewer chunks
)

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

# Function to handle file uploads and processing
def handle_upload(file):
    if file is None:
        return "No file uploaded."

    # Ensure uploads directory exists
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    # Save file to uploads directory
    filename = os.path.basename(file.name)
    filepath = os.path.join(upload_dir, filename)
    shutil.move(file.name, filepath)

    # Display processing message
    yield f"Processing '{filename}', please wait..."

    # Process file and upsert to database
    chunks = read_file(filepath, text_splitter)
    upsert(db, chunks, sparse_tokenizer, sparse_model, SPARSE_VECTOR_SIZE, dense_model, COLLECTION_NAME)

    # Display success message
    yield f"File '{filename}' uploaded and processed successfully."

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

def respond(message, retrieval_method, model, k, chat_history):
    # Display user's message immediately
    chat_history.append((message, ""))
    yield "", chat_history
    
    # Simulate the bot generating a response
    bot_response = get_model_output(
        client=db,
        query=message,
        retrieval_type=retrieval_method,
        top_k=k,
        model=model,
    )

    # Update the bot's response
    chat_history[-1] = (message, bot_response)
    yield "", chat_history


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
            choices=["hybrid", "sparse", "dense", "bm25", "tf-idf"], 
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
    
    # Chat interface without gr.ChatInterface for better control
    chatbot = gr.Chatbot(placeholder="What would you like to know?")
    msg = gr.Textbox(placeholder="Ask a question about the course...", show_label=False)
    send_btn = gr.Button("Send")

    send_btn.click(
        respond,
        inputs=[msg, retrieval_type, model_type, top_k, chatbot],
        outputs=[msg, chatbot]
    )

    msg.submit(
        respond,
        inputs=[msg, retrieval_type, model_type, top_k, chatbot],
        outputs=[msg, chatbot]
    )

    # File upload component
    with gr.Row():
        file_upload = gr.File(label="Upload a .txt file")

    upload_button = gr.Button("Process Upload")
    upload_message = gr.Markdown()

    upload_button.click(handle_upload, inputs=file_upload, outputs=upload_message)

    # Clear button to reset the chat
    clear = gr.Button("Clear")
    clear.click(lambda: ("", []), outputs=[msg, chatbot])
# Launch the app
if __name__ == "__main__":
    demo.launch(share=False)