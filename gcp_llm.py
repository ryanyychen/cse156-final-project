import google.generativeai as genai
import os

class GenAIStreamer:
    def __init__(self, api_key: str = os.environ["GOOGLE_CLOUD_API_KEY"], model_name: str = 'gemini-1.5-pro-latest'):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def stream_response(self, prompt: str):
        response = self.model.generate_content(prompt, stream=True)
        for message in response:
            yield message.text
    
    def get_response(self, prompt: str):
        response = self.model.generate_content(prompt)
        return response

    @staticmethod
    def list_available_models():
        return list(genai.list_models())

''' -- SAMPLE USAGE --
streamer = GenAIStreamer()
prompt = "Hello"
for text in streamer.stream_response(prompt):
    print(text)
'''
