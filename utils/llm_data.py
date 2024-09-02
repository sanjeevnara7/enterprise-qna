from .llm_core import *

import os
import pandas as pd
import json

DATA_SOURCE_NAMES = ['Sales', 'Customer Support Tickets', 'Invoices', 'Resumes']
DATA_SOURCE_PATH = '../data-samples'

def _load_data(name, location=DATA_SOURCE_PATH):
    if name == 'Sales':
        df = pd.read_csv(os.path.join(location, 'sales_data_sample_clean.csv'))
    elif name == 'Customer Support Tickets':
        df = pd.read_csv(os.path.join(location, 'customer_support_tickets_clean.csv'))
    elif name == 'Invoices':
        df = pd.read_csv(os.path.join(location, 'customer_shopping_data_clean.csv'))
    else:
        df = pd.read_csv(os.path.join(location, 'Resume_clean.csv'))
        df = df.drop(columns=['Category', 'ID', 'Resume_html', 'Key_points'])
    return df

def _load_metadata(name, location=DATA_SOURCE_PATH):
    metadata = """"""
    if name == 'Sales':
        sales_df = pd.read_csv(os.path.join(location, 'sales_data_sample_clean.csv'), nrows=0)
        column_names = sales_df.columns.tolist()
        metadata += f"""Sales: {{
            type: DataFrame,
            variable_name: sales_df,
            columns: {column_names}
        }}"""
    elif name == 'Customer Support Tickets':
        cst_df = pd.read_csv(os.path.join(location, 'customer_support_tickets_clean.csv'), nrows=0)
        column_names = cst_df.columns.tolist()
        metadata += f"""Customer Support Tickets: {{
            type: DataFrame,
            variable_name: cst_df,
            columns: {column_names}
        }}"""
    elif name == 'Invoices':
        invoice_df = pd.read_csv(os.path.join(location, 'customer_shopping_data_clean.csv'), nrows=0)
        column_names = invoice_df.columns.tolist()
        metadata += f"""Invoices: {{
            type: DataFrame,
            variable_name: invoice_df,
            columns: {column_names}
        }}"""
    else:
        resume_df = pd.read_csv(os.path.join(location, 'Resume_clean.csv'), nrows=0)
        resume_df = resume_df.drop(columns=['Category', 'ID', 'Resume_html', 'Key_points'])
        column_names = resume_df.columns.tolist()
        metadata += f"""Resumes: {{
            type: DataFrame,
            variable_name: resume_df,
            columns: {column_names}
        }}"""
    return metadata

def setup_metadata(data_sources=DATA_SOURCE_NAMES, data_source_path=DATA_SOURCE_PATH):
    metadatas = []
    for name in data_sources:
        metadatas.append(_load_metadata(name, data_source_path))

    return metadatas

def query_sources(user_query, metadatas):
    structure = "data_sources: [#List data source name]"
    prompt = f"""
    Here is a message from the user: "{user_query}". Based on this message, your job is to decide which of the data sources is/are required to answer the query. If the data sources cannot be used to answer the question, return an empty list.
    Return your answer as a json with the following structure. Make sure to return a valid json. Only return the json, nothing more:
    {structure}
    The data sources and their description is given below:
    {metadatas}
    """
    chat = get_chat_template()
    chat.append({'role': 'user', 'content': prompt})
    response = call_model(chat, max_new_tokens=max_new_tokens, pad_token_id=pad_token_id)
    json_out = response[0]['generated_text'][-1]["content"]
    data_sources = json.loads(json_out)
    return data_sources['data_sources']

def query_engine(user_query, data_sources, metadatas):
    sess_meta = []
    for name in data_sources:
        for metadata in metadatas:
            if name in metadata:
                sess_meta.append(metadata)
    structure = """
        #List each data source and query
        [
            {
                "data_source": #Dataframe name,
                "query_type": #query or eval,
                "query_str" #the query string to be run in df.query()
            },
            ...
        ]
    """

    prompt = f"""
    Here is a message from the user: "{user_query}". Based on this message, your job is to write pandas dataframe query strings (either df.query() or df.eval()) that retrieves relevant information based on the dataframe sources. Pay attention to the dataframe columns and notes (if provided) to form the query. Your query must be valid and be executable as python code. Remember to try all possibilities like lowercase/uppercase for text.
    Return your answer as a json with the following structure. Make sure to return a valid json. Only return the json, nothing more:
    {structure}
    The data can be retrieved from the following dataframe source(s):
    {sess_meta}
    """
    chat = get_chat_template()
    chat.append({'role': 'user', 'content': prompt})
    response = call_model(chat, max_new_tokens=max_new_tokens, pad_token_id=pad_token_id)
    json_out = response[0]['generated_text'][-1]["content"]
    pd_queries = json.loads(json_out)
    return pd_queries

def build_data_queries(user_query):
    metadata = setup_metadata()
    data_sources = query_sources(user_query, metadata)
    query_engine(user_query, data_sources, metadata)
    
def run_query(query_info, data, data_source_name):
    if "query" in query_info["query_type"]:
        if data_source_name in query_info["query_str"]:
            query_info["query_str"] = query_info["query_str"].replace(data_source_name, 'data')
            #Literal eval
            out = eval(query_info["query_str"])
            out = data[out]
        else:    
            out = data.query(query_info["query_str"])
        if not out.empty:
            return out.to_json()
        else:
            return 'No information found.'
    else:
        out = data.eval()
        return out.to_json()

#Usage - unit test
if __name__ == '__main__':
    metadata = setup_metadata()
    print(metadata)
    query_sources('I want to know which university did John Appleseed attend for his undergrad.', metadata)
    query_sources('I want to know the status of my issue with the Roomba robot. The case id is 100023.', metadata)
    