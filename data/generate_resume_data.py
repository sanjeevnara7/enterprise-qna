import sys
sys.path.append('../utils')
from llm_core import *

import pandas as pd
from tqdm import tqdm

def get_model_out(messages):
<<<<<<< HEAD
    response = call_model(messages, max_new_tokens=max_new_tokens, pad_token_id=pad_token_id)
    return response[0]['generated_text'][-1]["content"]

def generate_resume_kp(resume_str):
    
    prompt = f"""Here is a Resume. 
    The purpose of this data is to use it to fine tune a llm. 
    Extract the most import key points: the Applicant's First Name, Last Name, Email, Phone Number and Address
    Focus only on the data that is given in the Resume. If any of the information is not provided in the resume, leave it blank.

    Follow this structure and put the key points in json format. Only return the json, nothing more:
    first_name: #applicant's first name
    last_name: #applicant's last name
    email: #applicant's email
    phone: #applicant's phone
    address: #applicant's address
    
    Resume:
    {resume_str}
    """
    chat = get_chat_template()
    chat.append({'role': 'user', 'content': prompt})
    response = get_model_out(chat)
    return response #will contain kp

def process_resume_data(resume_data_df):
    kps = []
    for idx, row in tqdm(resume_data_df.iterrows(), total=len(resume_data_df)):
        try:
            resume_str = row['Resume_str']
            kp = generate_resume_kp(resume_str)
            kps.append(kp)
        except Exception as e:
            print('An error occurred at row:', idx)
            print(e)
            kps.append('')
    
    resume_data_df['Key_points'] = kps

    return resume_data_df
=======
    response = call_model(messages, max_new_tokens=max_new_tokens)
    return response[0]['generated_text'][-1]["content"]

def generate_resume_kp(resume_str):
    pass

def process_resume_data(resume_data_df):
    pass
>>>>>>> 0988ef31cccdf652d6416186dd7c744f4e10932d
