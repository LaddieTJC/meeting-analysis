import streamlit as st
import spacy
import pandas as pd
import spacy
from spacy import displacy
import nltk
import openai
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time

st.set_page_config(layout="wide")
openai.api_key = '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'
nltk.download('punkt')

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("fabiochiu/t5-small-medium-title-generation")
    model = AutoModelForSeq2SeqLM.from_pretrained("fabiochiu/t5-small-medium-title-generation")
    return tokenizer,model

tokenizer, model = load_model()
prompt_dict = {
    'Topic Analysis': 'Extract topics from the meeting notes: ',
    'Action Items Extraction':'Extract action items from this meeting notes and return in an ordered list: ',
    'Sentiment Analysis': 'Do sentiment analysis and extract sentences that contribute to the sentiment on this meeting notes: ' 
}

def meetingNotesAnalysis(notes,type):
    if type != 'Title Generation':
        prompt = prompt_dict[type] + notes
        response = openai.Completion.create(
                    model = "text-davinci-003",
                    prompt = prompt,
                    max_tokens = 100)
        return type + " : " + response['choices'][0]['text']
    else: 
        inputs = tokenizer(notes, max_length=512, truncation=True, return_tensors="pt")
        output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=32)
        decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        predicted_title = nltk.sent_tokenize(decoded_output.strip())[0]
        return  type + " : " + predicted_title

with st.container():
    col1,col2 = st.columns(2,gap="large")
    with col1:
        input_text = st.text_area(label="Input your meeting notes", height=300)
        options = st.multiselect("Pick Analysis:",['Topic Analysis','Title Generation','Action Items Extraction','Sentiment Analysis'])
        userInput = st.text_input('OR Enter your own type of analysis:')
        submit = st.button("Submit")
            
    if submit:
        if input_text and options:
            with col2:
                st.markdown("You selected: "+", ".join(options))

                for i in options:
                    st.markdown(meetingNotesAnalysis(input_text,type=i))

                # inputs = ["summarize: " + input_text]
                # inputs = tokenizer(inputs, max_length=512, truncation=True, return_tensors="pt")
                # output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=32)
                # decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
                # predicted_title = nltk.sent_tokenize(decoded_output.strip())[0]
                # st.markdown(":blue[Title Generation]:  " + predicted_title)
                # action_items = meetingNotesAnalysis(input_text,type="Action Items Extraction")
                # st.write("Action Item sentences:\n",action_items)
                # sentiment = meetingNotesAnalysis(input_text,type='Sentiment Analysis')
                # st.write(sentiment)
        if input_text and userInput:
                with col2:
                # st.write("prompt = ",userInput + " : " + input_text)
                    prompt = userInput + " : " + input_text
                    response = openai.Completion.create(
                        model = "text-davinci-003",
                        prompt = prompt,
                        max_tokens = 100)        
                    
                    st.markdown(":blue[Custom Analysis]: "+ response['choices'][0]['text']) 
                        

