import sys
sys.path.append('../utils')

from llm_core import *

from pprint import pprint
import time
import random
import datetime
import json
import pandas as pd
from tqdm import tqdm

date_time = datetime.datetime.now()
d = date_time.strftime("%s")

#Set environment variable with API Key before usage

def get_model_out(messages):
    response = call_model(messages, max_new_tokens=2048)
    return response[0]['generated_text'][-1]["content"]

def create_random_prompt(sales_data_row, roles=["Customer", "Salesman"], range_vals=(3, 15), industries=None):
    if industries is None:
        industries = ["tech", "health", "finance"]  # default industries; replace with your default list if different
    
    x = random.randint(*range_vals)

    conversation_structure = ""
    for i in range(1, x+1):
            conversation_structure += f"""
        {roles[0]}: #{i}. sentence of {roles[0].lower()}
        {roles[1]}: #{i}. sentence of {roles[1].lower()}"""

    prompt = f"""Here is a row from a Sales Dataset. 
    The purpose of this data is to use it to fine tune a llm. 
    Generate a conversation example that is based on the data that is provided and would help an ai to learn the sales by examples. 
    Focus only on the data that is given in the row when generating the examples.  Pay attention to the customer name to determine the industry/domain, and let the conversation be in that particular industry.

    Follow this structure and put the conversation in json format. Only return the json, nothing more:
    {conversation_structure}

    Sales Data:
    {sales_data_row}
    """

    return prompt

def generate_sales_data_row(sales_data_row):
    prompt = create_random_prompt(sales_data_row)
    chat = get_chat_template()
    chat.append({'role': 'user', 'content': prompt})
    response = call_model(chat, max_new_tokens=max_new_tokens)
    synthetic_conv = response[0]['generated_text'][-1]["content"]
    return synthetic_conv

def generate_sales_data(sales_df):
    convs = []
    for idx, row in tqdm(sales_df.iterrows()):
        try:
            conv = generate_sales_data_row(row)
            convs.append(conv)
        except Exception as e:
            print('An error occurred at row:', idx)
            print(e)
            convs.append('')

    sales_df['CONVERSATION'] = convs
    
    return sales_df
    