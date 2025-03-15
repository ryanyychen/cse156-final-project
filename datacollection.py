# Data collection code (run this as a script externally)
from google.oauth2 import service_account
from googleapiclient.discovery import build
import re
import time
import os

SCOPES = ['https://www.googleapis.com/auth/documents.readonly']

creds = service_account.Credentials.from_service_account_file('credentials.json', scopes=SCOPES)
docs_service = build('docs', 'v1', credentials=creds)

doc_id = ['1qkv3QsNnd615xWlK-4FD0uksfxgzFTwS5Yfk3zWX5UA', '1oGk5m8ISfjg7Qwhy7vjpJsiA611zVGZ0AyqUwOqnYSA', '1phL5UoMFg8tM42lD31GvSB1OuzvKeAaGPBeTImEGsYI',
          '1X1zna-BvvemEBLuMecPoxFElSVgkXSM1KouywfHdnkA', '1YT5Py9qAWlD2q-naZ6e0Z5tdRomKqQiCQoOpbNU5Lr0', '18Xlij4IqO-HyKEyIJA_0lx0oYuoIWudNTRRUAWuZFKw',
          '1ulJZWbD-eknCx-NwU2TPsqyNe6v2s6OMhZa493iEj14', '12dGj-vozjfEorTthV0AEMsqUVy45SsicLJIrSGBYldA', '1wqDoolUWZSnC0dvdIDArSHP2LLRXSIYVDFTx2R_HxXE',
          '1UocQFq4KFCq_Ma0qAKkLqV59g7mi-JmbRUSJwbSda7E', '1xlgxJlGkC3Tv7dXPSw22txGFEIA7FVMU087OsHByhE4', '18G2ZZCg4TRXnrxRXWcx1ahQVi34lcNFnVGRuk9AnNl4',
          '14lSZNHg8YNPv1hFcy_UTn9cfvuYoQlLu1TTAhkDW_1c', '1Sinu4Dm_xGS4UhfGGGQt2LwCJrf8BRMVRcFBcmOXTTM', '1Fm76hz-3MjdznyNkpWmXJkcDb0OMZmzGZNjgk_7jhak',
          '1NTQESxuMbxAD4858omcWjbiNwmlVCbQxVMH8ReQfxkU', '11ix5t4PPeVuTCJFNe9yzj4_1NqSQDH2i-ULB_C9VLbM']

def extract_all_text_from_google_doc(doc_id):
    document = docs_service.documents().get(documentId=doc_id).execute()
    text_content = []

    for content in document.get('body', {}).get('content', []):
        if 'paragraph' in content:
            for element in content['paragraph'].get('elements', []):
                if 'textRun' in element:
                    text_content.append(element['textRun']['content'])

    return ''.join(text_content)

def savetxt(title, text):
    folder = "lectureslides"
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, title)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)

for i in range(1, len(doc_id)+1):
    doc_text = extract_all_text_from_google_doc(doc_id[i-1])
    title = "cse156Lecture" + str(i) + "Slides.txt"
    savetxt(title, doc_text)
    time.sleep(1)
    print("File saved " + str(i))
