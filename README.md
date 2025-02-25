# cse156-final-project

# Data Collection
### Use Google Docs API to extract text from class lectures
### Steps to run

1. Create a new project on Google Cloud Console. Then under the **APIs and Services** section, select **Enable APIs and Services**, and search/enable the **Google Docs API**.
2. On the Google Cloud Console, navigate to **IAM & Admin > Service Accounts** and create a service account (any name is fine). Set the role to owner.
3. On the Google Clout Console, click on the **Service Account name** from step 2, and create a new key with **JSON** as the key type. Make sure to include this file as "credentials.json" in the same folder as the script below.
4. Before running the script, make sure to make a copy of the **Google Doc** version of the lecture slides, **not the Google Drive PDF version**. Then, share the document with the email address found in the **credentials.json** file, which should be in the form of `[service_account_name]@[project_name].iam.gserviceaccount.com`.
5. Finally, run this script (preferrably locally). The included doc_ids are what my file ids are, but yours would be different. This would be in the Google Doc url after `https://docs.google.com/document/d/` and before `/edit?tab=t.0`
