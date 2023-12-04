import streamlit as st
from streamlit_chat import message
import json
from pathlib import Path
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
os.environ["OPENAI_API_KEY"] = "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"
from openai import OpenAI

genmed = 'GenMedGPT.json'
meddia = 'MedDialog.json'
embeddings = HuggingFaceEmbeddings(model_name="bge-small-en-v1.5")
db = FAISS.load_local("faiss_index", embeddings)

def RAG(input,k=5):
    ret_docs = db.max_marginal_relevance_search(input,k)
    rag = []
    for doc in ret_docs:
        if doc.metadata["source"] == genmed:
            context = json.loads(Path(genmed).read_text())[doc.metadata['seq_num']]
            input = context["input"]
            output = context["output"]
            rag.append(f"patient: {input}\ndoctor: {output}")
        if doc.metadata["source"] == meddia:
            context = "\n".join(json.loads(Path(meddia).read_text())[doc.metadata['seq_num']]['utterances'])
            rag.append(f"{context}")
    rag = "\n\n".join(rag)
    return rag


def prompt_template(prompt):
    rag = RAG(prompt)
    prompt_ = f'''system message: you are a doctor who provide solution to patient's symptom.
Here is some relevant talk between doctors and patients.
####
{rag}
####
user message: patient: {prompt}.
Assistant role: doctor: '''
    return prompt_
    

def generate_response(prompt):
  '''
  This function input a prompt, call the api of openai and response.
  '''
  client = OpenAI()
  completion = client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=[
          {"role": "system", "content": "you are an expert doctor, Waston."},
          {"role": "user", "content": prompt}
        ],
          temperature=1e-7,
          presence_penalty =1.1,
          top_p =1
      )
  return completion.choices[0].message.content

st.title("Expert Clinic")
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
user_input=st.text_input("You:",key='input')
if user_input:
    prompt = prompt_template(user_input)
    output=generate_response(prompt)
    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(output)
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))
  
