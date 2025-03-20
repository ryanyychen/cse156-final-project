import torch
import numpy as np
from dotenv import load_dotenv
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from qdrant_client.models import SparseVector
from embeddings import vectorize

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

load_dotenv()

#### Context Retrieval ####
'''
Normalize scores to be between 0 and 1

Args:
    scores: numpy array of scores
Returns:
    numpy array of normalized scores
'''
# Assume sparse_scores and dense_scores are numpy arrays
def normalize(results):
    scores = [point.score for point in results.points]
    min_score = np.min(scores)
    max_score = np.max(scores)
    for point in results.points:
        point.score = (point.score - min_score) / (max_score - min_score)
    return results

'''
Perform a similarity search on sparse embeddings

Args:
    client (QdrantClient): client object with connection to database
    query_sparse: sparse embedding of query
    top_k: number of top results to return
    collection_name: Name of the collection to use
Returns:
    top_k results from similarity search on sparse embeddings
'''
def sparse_query(client, query_sparse, top_k, collection_name):
    results = client.query_points(
        collection_name=collection_name,
        query=SparseVector(
            indices=query_sparse["indices"],
            values=query_sparse["values"],
        ),
        using="sparse_vector",
        limit=top_k,
    )
    return normalize(results)

'''
Perform a similarity search on dense embeddings

Args:
    client (QdrantClient): client object with connection to database
    query_dense: dense embedding of query
    top_k: number of top results to return
    collection_name: Name of the collection to use
Returns:
    top_k results from similarity search on dense embeddings
'''
def dense_query(client, query_dense, top_k, collection_name):
    results = client.query_points(
        collection_name=collection_name,
        query=query_dense,
        using="dense_vector",
        limit=top_k,
    )
    return normalize(results)

'''
Calculates a weighted score for each query result

Args:
    sparse_score: similarity score of sparse embedding
    dense_score: similarity score of dense embedding
Returns:
    weighted score combining both sparse and dense scores
'''
def weighted_score(sparse_score, dense_score, sparse_weight=0.3, dense_weight=0.7):
    return (sparse_weight * sparse_score) + (dense_weight * dense_score)

'''
Combines sparse and dense query results

Args:
    sparse_results: results from sparse query
    dense_results: results from dense query
Returns:
    combined results from sparse and dense queries
'''
def combine_queries(sparse_results, dense_results):
    # Gather scores for all results
    all_results = {}
    for point in dense_results.points:
        all_results[point.id] = {"dense_score": point.score, "sparse_score": 0, "text": point.payload["text"]}

    for point in sparse_results.points:
        if point.id not in all_results:
            all_results[point.id] = {"dense_score": 0, "sparse_score": point.score, "text": point.payload["text"]}
        else:
            all_results[point.id]["sparse_score"] = point.score

    # Weighted results
    scored_results = []
    for result in all_results.keys():
        sparse_score = all_results[result]["sparse_score"]
        dense_score = all_results[result]["dense_score"]
        scored_results.append({
            "score": weighted_score(sparse_score, dense_score, sparse_weight=0.3, dense_weight=0.7),
            "text": all_results[result]["text"],
        })

    return scored_results

'''
Perform a hybrid query (sparse and dense) on vector database to provide as context to llm

Args:
    client (QdrantClient): client object with connection to database
    query (String): input from user asked to LLM
    top_k: number of top results to return
    sparse_tokenizer: Tokenizer for sparse embeddings
    sparse_model: Model for sparse embeddings
    sparse_vector_size: Size of sparse vectors
    dense_model: Model for dense embeddings
    collection_name: Name of the collection to use
Returns:
    list of text to serve as context for LLM
'''
def hybrid_query(client, query, top_k, sparse_tokenizer, sparse_model, sparse_vector_size, dense_model, collection_name):
    # Vectorize the query
    query_vector = vectorize(query, sparse_tokenizer, sparse_model, sparse_vector_size, dense_model, prefix="query: ")
    query_sparse = query_vector[0]
    query_dense = query_vector[1].cpu().numpy()

    # Query database using both sparse and dense embeddings
    sparse_results = sparse_query(client, query_sparse, top_k, collection_name)
    dense_results = dense_query(client, query_dense, top_k, collection_name)

    # Combine and calculate weighted scores for all results
    scored_results = combine_queries(sparse_results, dense_results)

    # Sort in descending order by combined score
    sorted_results = sorted(scored_results, key=lambda x: x["score"], reverse=True)

    # Trim to top k results
    sorted_results = sorted_results[:top_k]

    # Return results
    return sorted_results

"""
Retrieves documents from the Qdrant vector database.

Args:
    client: QdrantClient instance.
    collection_name (str): Name of the collection.
    limit (int): Maximum number of documents to retrieve.

Returns:
    list: List of document texts.
"""
def retrieve_documents_from_qdrant(client, collection_name):
    # Fetch points from the collection using scroll
    search_result = client.scroll(collection_name=collection_name, limit=2500)
    count = client.count(collection_name=collection_name)

    # The first element in the tuple is the actual list of points
    points = search_result[0]  # Extract the list of points

    documents = []
    for point in points:
        doc_text = point.payload.get("text", "")
        documents.append(doc_text)
    return documents

'''
Perform a bm25 query on tokenized documents to provide as context to llm

Args:
    client (QdrantClient): client object with connection to database
    query (String): input from user asked to LLM
    top_k: number of top results to return
    collection_name: Name of the collection to use
Returns:
    list of text to serve as context for LLM
'''
def bm25_query(client, query, collection_name, top_k=5):
    documents = retrieve_documents_from_qdrant(client, collection_name=collection_name)

    tokenized_documents = [word_tokenize(doc.lower()) for doc in documents]

    bm25 = BM25Okapi(tokenized_documents)

    tokenized_query = word_tokenize(query.lower())

    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    sorted_results = [documents[i] for i in ranked_indices[:top_k]]

    return sorted_results