import sys
sys.path.append('../utils')
from llm_core import *

import pandas as pd
from tqdm import tqdm

def get_model_out(messages):
    response = call_model(messages, max_new_tokens=max_new_tokens)
    return response[0]['generated_text'][-1]["content"]

def generate_resume_kp(resume_str):
    pass

def process_resume_data(resume_data_df):
    pass