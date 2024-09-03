## Enterprise Question-Answering System using Agentic LLMs/multimodal RAG
<img src="https://img.shields.io/badge/python-3.10-green" /> <img src="https://img.shields.io/badge/transformers-4.44.2-yellow" /> <img src="https://img.shields.io/badge/semantic--router-0.0.20-blue"/><br>

An implementation of a question-answering system using LLaMA-1.3-8B-Instruct with sample enterprise data.

<p align="center">
  <img src="https://github.com/sanjeevnara7/enterprise-qna/blob/main/assets/erp_flowchart.png" width="50%">
</p>

## System Overview
The system implemented in this repo is a minimal version of the system in the diagram.
- Data Ingestion Layer: Retrieves data from .csv files downloaded from Kaggle and preprocesses them for retrieval during runtime.
- Semantic Routing: Convert the input message to a semantic vector and pass it through a graph of
semantic routes to determine the intent of the user.
- Query and Retrieval Engine: Determine what kind of data is required, and retrieve it from the
database. The relevant data will be appended as context to the user’s message.
- LLM Prediction : The center of the system which will generate answers for the users queries. Uses LLaMA-1.3-8B-Instruct as the underlying model.

## Data
This Q&A system uses the following sample data sources, all downloaded from Kaggle:
- Sales data
- Invoice data
- Customer Support Ticket data
- Resume data

Each dataset was downloaded, cleaned, and randomly subsampled to created small versions of the data. Unstructured data, such as Sales Conversations, have been synthetically generated using LLaMA and added to the datasets to facilitate cross-queries that span multiple data sources.

Check `data/data_description.ipynb` to get a better understanding of each of the datasets that have been downloaded.

## Usage

1. The model used is LLaMA-1.3-8B which requires sign-up on HuggingFace. It is also recommended to use a GPU with at least 20GB of VRAM. Check `utils/llm_core` for more details.
    
2. Run `git clone https://github.com/sanjeevnara7/enterprise-qna.git` to clone this repository.
  
3. `pip install -e .` to install the necessary packages.
  
4. Import and initialize the Chatbot class:
  ```python
  import utils.llm_chat
  
  chatbot = utils.llm_chat.Chatbot('./data-samples')
  ```
5. Start chatting:
```python
message = 'Hey, how are you?'
print("user", ':')
print(message)
print('-'*20)

_, history = chatbot.chat(message)
item = history[-1]
print(item["role"], ':')
print(item["content"])
print('-'*20)
```
```
Output:
>>>user :
>>>Hey, how are you?
>>>--------------------
>>>assistant :
>>>I'm doing well, thank you for asking. I'm a large language model, so I don't have feelings in the same way humans do, but I'm functioning properly and ready to help you with any questions or tasks you may have. How about you? How's your day >>>going so far?
>>>--------------------
```

### Examples:
```
Output:
>>>user :
>>>I want to know where did Kpandipou study. Tell me about his education history.
>>>--------------------
>>>assistant :
>>>Based on the information I got, Kpandipou's education history is as follows:
>>>
>>>1. **Associate of Arts**: Suffolk University (Jun 2007) - Major in International Business and Business Administration.
>>>
>>>2. **BBA (Bachelor of Business Administration)**: Golden Gate University (Jun 2011) - Major in Business Administration, Minor in International Business. He was an International Student Scholarship Recipient.
>>>
>>>3. **MBA (Master of Business Administration)**: IGlobal University (Jun 2015) - Major in Marketing.
>>>--------------------
```
```
Output:
>>>user :
>>>I want to know the status of my issue with the Roomba robot. The case id is 7076.
>>>--------------------
>>>assistant :
>>>Based on the information provided, your case ID is 7076 and the ticket status is "Pending Customer Response." 
>>>
>>>This means that the support team has not received a response from you yet, and they are waiting for your input to proceed with resolving your issue. 
>>>
>>>To get an update on your issue, I would recommend reaching out to the support team and providing any necessary information or clarification they may need. 
>>>
>>>Additionally, since your ticket priority is "High," it's recommended that you respond to the support team as soon as possible to ensure a timely resolution to your issue.
>>>--------------------
```

### Known Issues
- Invalid JSON responses from model that results in exceptions when trying to parse the JSON. This usually occurs when the retrieval query that is required is too complex and cannot be formed well by the model.
