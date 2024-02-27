from mistralai.client import MistralClient, ChatMessage
#import numpy as np
#import faiss
import os

apikey=os.environ.get('LECHATBLEUAPIKEY')
client = MistralClient(endpoint="https://lechat-bleu-serverless.francecentral.inference.ai.azure.com",
                       api_key=apikey)

def run_mistral(user_message):
    #client = MistralClient(api_key=api_key)
    messages = [
        ChatMessage(role="user", content=user_message)
    ]
    chat_response = client.chat(
        model="azureai",
       
        messages=messages
    )

    return (chat_response.choices[0].message.content)
def run_stream_mistral(user_message):    
    for chunk in client.chat_stream(
        model="azureai",
        messages=[ChatMessage(role="user", content=user_message)],
    ):
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")
    return
# def get_text_embedding(input):
#     embeddings_batch_response = client.embeddings(
#           model="azureai",
#           input=input
#       )
#     return embeddings_batch_response.data[0].embedding

#print(run_mistral("Quelles sont les spécialités culinaire de Versailles ?"))
print(run_stream_mistral("Quelles sont les spécialités culinaire de Versailles ?"))

#print(get_text_embedding("Quelles sont les spécialités culinaire de Versailles ?"))