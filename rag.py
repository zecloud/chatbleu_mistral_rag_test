import os
import requests
import faiss
import numpy as np


from mistralai.client import MistralClient, ChatMessage

import os

apikey=os.environ.get('LECHATBLEUAPIKEY')
client = MistralClient(endpoint="https://lechat-bleu-serverless.francecentral.inference.ai.azure.com",
                       api_key=apikey)

def run_mistral(user_message):
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


apikey= os.environ.get("BAGUETTEMBEDDINGAPIKEY")


from langchain_openai import AzureOpenAIEmbeddings

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint="https://baguettegpt.openai.azure.com",
    api_key=apikey,
    azure_deployment="baguettembedding",
    openai_api_version="2023-05-15",
)

response = requests.get('https://raw.githubusercontent.com/zecloud/chatbleu_test/main/textaymeric.txt')
text = response.text
chunk_size = 2048
chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

doc_result = embeddings.embed_documents([text])

print(doc_result)
text_embeddings=np.array(doc_result)
print(text_embeddings.shape)
d = text_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(text_embeddings)

question = "Qui est Aymeric Weinbach ?"
question_embeddings = np.array([embeddings.embed_query(question)])
question_embeddings.shape
print(question_embeddings)

D, I = index.search(question_embeddings, k=2)
print(I)

retrieved_chunk = [chunks[i] for i in I.tolist()[0]]
#print(retrieved_chunk)

prompt = f"""
Les informations de contexte ci-dessous .
---------------------
{retrieved_chunk}
---------------------
Au vu de ces informations de contexte et sans connaissances préalables, répondez à la requête.
Requete: {question}
Réponse:
"""

print(run_stream_mistral(prompt))
#query_result = embeddings.embed_query(text)