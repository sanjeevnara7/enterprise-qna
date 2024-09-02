'''
<<<<<<< HEAD
This file contains core LLM utilities to allow chat with the model
=======
This file contains core LLM functionalities to allow chat with the model
>>>>>>> 0988ef31cccdf652d6416186dd7c744f4e10932d
'''

import os
from transformers import pipeline
from pprint import pprint

#HF cache
os.environ['HF_HOME'] = './.hf_cache'

#Chat settings
device = 'cuda:0' #Set this according to your device
max_new_tokens = 1024
def get_chat_template():
    chat_template = [
        {"role": "system", "content": "You are a helpful AI Enterprise Assistant. Your job is to provide polite answers to the user's questions."},
    ]
    return chat_template

#export auth token os environment variable before using
#hf_token = os.environ['HF_AUTH_TOKEN']
pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3.1-8B-Instruct", device=device) #token=hf_token)
<<<<<<< HEAD
pad_token_id = pipe.tokenizer.eos_token_id
=======
>>>>>>> 0988ef31cccdf652d6416186dd7c744f4e10932d

def call_model(*args, **kwargs):
    if args and kwargs:
        out = pipe(*args, **kwargs)
    elif args:
        out = pipe(*args)
    else:
        out = pipe(**kwargs)
    return out

def show_chat_pretty(chat):
    print('*'*40)
    for item in chat:
        print(item["role"], ':')
        print(item["content"])
        print('-'*20)
    print('*'*40)
<<<<<<< HEAD
    
=======
>>>>>>> 0988ef31cccdf652d6416186dd7c744f4e10932d

if __name__ == "__main__":
    #Unit test
    chat = get_chat_template()
    chat.append({"role": "user", "content": "Hey, can you tell me about Salesforce and why it is useful?"})
    response = call_model(chat, max_new_tokens=max_new_tokens)
    show_chat_pretty(response[0]['generated_text'])
