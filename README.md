# Data Collection
### Use Google Docs API to extract text from class lectures
### Steps to run

1. Create a new project on Google Cloud Console. Then under the **APIs and Services** section, select **Enable APIs and Services**, and search/enable the **Google Docs API**.
2. On the Google Cloud Console, navigate to **IAM & Admin > Service Accounts** and create a service account (any name is fine). Set the role to owner.
3. On the Google Clout Console, click on the **Service Account name** from step 2, and create a new key with **JSON** as the key type. Make sure to include this file as "credentials.json" in the same folder as the script below.
4. Before running the script, make sure to make a copy of the **Google Doc** version of the lecture slides, **not the Google Drive PDF version**. Then, share the document with the email address found in the **credentials.json** file, which should be in the form of `[service_account_name]@[project_name].iam.gserviceaccount.com`.
5. Finally, run this script (preferrably locally). The included doc_ids are what my file ids are, but yours would be different. This would be in the Google Doc url after `https://docs.google.com/document/d/` and before `/edit?tab=t.0`

# Initializing the RAG system

Make sure you have in your `.env` file an API key to Qdrant under the variable name `QDRANT_API_KEY`, and update `COLLECTION_NAME` to the collection where you want the vectors to be stored.

### Initialize the sparse and dense embedding models:
- Sparse (TF-IDF)
- Sparse (BM25)
- Sparse (SPLADE): set `SPARSE_MODEL_NAME = "naver/splade_v2_distil"`
- Hybrid (SPLADE + E5): set `SPARSE_MODEL_NAME = "naver/splade_v2_distil"` and `DENSE_MODEL_NAME = "intfloat/e5-base"`

### Initializing the Qdrant database
- Put all `txt` files you wish to add as context in the `documents` directory.
- Then run the cell titled `Main` to process all files and upload them to the vector database.

### Querying the Qdrant database
- You can change the query by editing the string
- Results of sparse and dense queries are returned as QueryResponse, access each response point using `QueryResponse.points`
- Results of hybrid query are returned as a list of dictionaries with keys `score`, `text`