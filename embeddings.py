#### Imports ####
import torch
from dotenv import load_dotenv

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

load_dotenv()

#### Embeddings and Vectorizing ####
'''
Sparse embeddings using given model

Args:
    text (String): string to be embedded
    sparse_tokenizer: Tokenizer for sparse embeddings
    sparse_model: Model for sparse embeddings
    sparse_vector_size: Size of sparse vectors
    threshold: Threshold for sparse embedding
Returns:
    sparse embedding of text
'''
def sparse_embed(text, sparse_tokenizer, sparse_model, sparse_vector_size, threshold=0.1, top_k=500):
    tokenized_text = sparse_tokenizer(
        text,
        truncation=False,
        return_tensors="pt",
    ).to(DEVICE)

    input_ids = tokenized_text['input_ids'].squeeze(0)
    total_length = len(input_ids)

    # Split tokens into chunks of 512
    chunk_size = sparse_model.config.max_position_embeddings
    chunks = [input_ids[i:i+chunk_size] for i in range(0, total_length, chunk_size)]

    # Get embeddings for each chunk
    embeddings = []
    for chunk in chunks:
        chunk_input = {'input_ids': chunk.unsqueeze(0)}

        with torch.no_grad():
            outputs = sparse_model(**chunk_input)

        # Aggregate embeddings using mean pooling across token positions
        sparse_embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)

        # Apply ReLU and threshold to enforce sparsity
        sparse_embedding = torch.relu(sparse_embedding)
        sparse_embedding[sparse_embedding < threshold] = 0

        # Get non-zero indices and corresponding values
        non_zero_indices = sparse_embedding.nonzero(as_tuple=True)[0]
        non_zero_values = sparse_embedding[non_zero_indices]

        # Ensure indices fall within the sparse_vector_size
        mask = non_zero_indices < sparse_vector_size
        valid_indices = non_zero_indices[mask]
        valid_values = non_zero_values[mask]

        # Append to embeddings list
        embeddings.append((valid_indices, valid_values))

    # Concatenate all embeddings
    concatenated_indices = []
    concatenated_values = []
    for indices, values in embeddings:
        concatenated_indices.extend(indices)
        concatenated_values.extend(values)

    # Keep only the top-k most important dimensions for efficiency
    if len(concatenated_indices) > top_k:
        top_k_indices = torch.topk(torch.tensor(concatenated_values), k=top_k).indices
        concatenated_indices = [concatenated_indices[i] for i in top_k_indices]
        concatenated_values = [concatenated_values[i] for i in top_k_indices]

    # Move to CPU and convert to lists for Qdrant compatibility
    return {
        "indices": valid_indices.cpu().numpy().tolist(),
        "values": valid_values.cpu().numpy().tolist()
    }

'''
Dense embeddings

Args:
    text (String): string to be embedded
    dense_model: Model for dense embeddings
Returns:
    dense embedding of text
'''
def dense_embed(text, dense_model, prefix="passage: "):
    # Format required for E5
    formatted_text = prefix + text
    return torch.tensor(dense_model.encode(formatted_text, normalize_embeddings=True)).to(DEVICE)

'''
Generate both sparse and dense vectors for text

Args:
    text (String): string to be embedded
    sparse_tokenizer: Tokenizer for sparse embeddings
    sparse_model: Model for sparse embeddings
    sparse_vector_size: Size of sparse vectors
    dense_model: Model for dense embeddings
Returns:
    tuple of (sparse_embedding, dense_embedding)
'''
def vectorize(text, sparse_tokenizer, sparse_model, sparse_vector_size, dense_model, prefix="passage: "):
    # Get sparse embedding with vocabulary-sized vector
    sparse_embedding = sparse_embed(text, sparse_tokenizer, sparse_model, sparse_vector_size)

    # Get dense embedding
    dense_embedding = dense_embed(text, dense_model, prefix=prefix)

    return (sparse_embedding, dense_embedding)