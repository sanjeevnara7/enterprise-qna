'''
This file contains the implementation of the chat system for question-answering.
'''
from . import llm_core
from . import llm_data
from . import semantic_routes

import os
import json
import pandas as pd

import traceback

class Chatbot():
    def __init__(self, data_path, max_chat_messages=4): #Limited to 4 due to GPU constraints
        self.pipe = llm_core.pipe
        self.template = llm_core.get_chat_template()
        self.device = llm_core.device
        self.max_new_tokens = llm_core.max_new_tokens
        self.pad_token_id = llm_core.pad_token_id
        self.data_path = data_path
        self.max_chat_messages = max_chat_messages
        self.context_data = None #session context
        self.setup_data_sources()
        self.setup_router()
        self.show_init_message()

    def setup_data_sources(self):
        '''Load data into memory'''
        self.metadatas = llm_data.setup_metadata(data_source_path=self.data_path)
        self.sales_df = llm_data._load_data('Sales', location=self.data_path)
        self.cst_df = llm_data._load_data('Customer Support Tickets', location=self.data_path)
        self.invoice_df = llm_data._load_data('Invoices', location=self.data_path)
        self.resume_df = llm_data._load_data('Resumes', location=self.data_path)

    def setup_router(self):
        '''
        Setup Router for semantic routing
        '''
        self.router = semantic_routes.init_route_layer()

    def show_init_message(self):
        print('*'*20, 'Enterprise Q&A System with LLaMA-3.1-8B-Instruct', '*'*20)
        print('[*] The chat system has been initialized. The available data sources are listed below. If you have any questions or would like to know about specific information, please enter your message.\n\n')
        for item in self.metadatas:
            print(item)
        print('*'*60)

    def call_model(self, *args, **kwargs):
        if args and kwargs:
            out = self.pipe(*args, **kwargs)
        elif args:
            out = self.pipe(*args)
        else:
            out = self.pipe(**kwargs)
        return out

    def chat(self, message, chat_history=None):
        if chat_history is None:
            chat_history = self.template
        if len(chat_history) > self.max_chat_messages + 1:
            chat_history = chat_history[:1] + chat_history[-self.max_chat_messages:] #System prompt + remainder
        
        #Perform semantic routing
        route = self.router(message)
        if route.name == 'chitchat':
            #chitchat doesn't require data retrieval
            chat_history.append({"role": "user", "content": message})
            response = self.call_model(chat_history, max_new_tokens=self.max_new_tokens, pad_token_id=self.pad_token_id)
            gen_out = response[0]['generated_text'][-1]["content"]
            chat_history.append(response[0]['generated_text'][-1])
            return gen_out, chat_history

        #Retrieve data
        data_sources = llm_data.query_sources(message, self.metadatas)
        print(data_sources)
        if data_sources is None or len(data_sources) == 0 and route.name is None:
            
            chat_history.append({"role": "user", "content": message})
            response = self.call_model(chat_history, max_new_tokens=self.max_new_tokens, pad_token_id=self.pad_token_id)
            gen_out = response[0]['generated_text'][-1]["content"]
            chat_history.append(response[0]['generated_text'][-1])
            return gen_out, chat_history
        
        #print('data sources:',data_sources)
            
        #Perform data queries
        pd_queries = llm_data.query_engine(message, data_sources, self.metadatas)
        print(pd_queries)
        context_data = self.context_data or []
        for query in pd_queries:
            try:
                # print(query["data_source"])
                data = getattr(self, query["data_source"], None)
                data_res = llm_data.run_query(query, data, query["data_source"])
                context_data.append(data_res)
            except Exception:
                # print(traceback.format_exc())
                context_data.append('Sorry, there was no relevant information found for that.')

        if len(context_data) == 0:
            context_data = 'No additional context data available.'
        else:
            self.context_data = context_data
        
        message += f"""
        You have previously retrieved the following information as additional context. Use this information to answer. Do not mention that you have used additional information:
        {context_data}
        """
        chat_history.append({"role": "user", "content": message})
        response = self.call_model(chat_history, max_new_tokens=self.max_new_tokens, pad_token_id=self.pad_token_id)
        gen_out = response[0]['generated_text'][-1]["content"]
        chat_history.append(response[0]['generated_text'][-1])
        return gen_out, chat_history

#Unit test
if __name__ == "__main__":
    chatbot = Chatbot('../data-samples')
    gen_out, chat_history = chatbot.chat('I want to know where did Kpandipou study. Could you give me their education history?')
    llm_core.show_chat_pretty(chat_history)