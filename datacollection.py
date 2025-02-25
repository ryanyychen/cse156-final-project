# Data collection code (run this as a script externally)
from google.oauth2 import service_account
from googleapiclient.discovery import build
import re
import time

SCOPES = ['https://www.googleapis.com/auth/documents.readonly']

creds = service_account.Credentials.from_service_account_file('credentials.json', scopes=SCOPES)
docs_service = build('docs', 'v1', credentials=creds)

doc_id = ['1qkv3QsNnd615xWlK-4FD0uksfxgzFTwS5Yfk3zWX5UA', '1oGk5m8ISfjg7Qwhy7vjpJsiA611zVGZ0AyqUwOqnYSA', '1phL5UoMFg8tM42lD31GvSB1OuzvKeAaGPBeTImEGsYI',
          '1X1zna-BvvemEBLuMecPoxFElSVgkXSM1KouywfHdnkA', '1YT5Py9qAWlD2q-naZ6e0Z5tdRomKqQiCQoOpbNU5Lr0']

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
    clean = re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', '', text)
    with open(title, 'w', encoding='utf-8') as file:
        file.write(clean)

for i in range(1, len(doc_id)+1):
    doc_text = extract_all_text_from_google_doc(doc_id[i-1])
    title = "cse156Lecture" + str(i) + "Slides.txt"
    savetxt(title, doc_text)
    time.sleep(1)
    print("File saved " + str(i))
