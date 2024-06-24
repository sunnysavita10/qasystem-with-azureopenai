import os
from openai import AzureOpenAI
import json


client = AzureOpenAI(
    api_version="2024-02-01",
    azure_endpoint="https://mychatmodel.openai.azure.com/",
    api_key=""
)

completion = client.chat.completions.create(
    model="chatmodel",
    messages=[
        {
            "role": "user",
            "content": "what is a meaning of life can you explain in nutshell?",
        }
    ]
)
      
print(completion.to_json()) 