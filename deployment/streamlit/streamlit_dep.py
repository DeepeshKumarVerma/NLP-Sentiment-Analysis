import streamlit as st
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import numpy as np
import zipfile 
import os

# O. Unzip the model folder if not already unzipped
zip_path= r"~\BERT\BERT_sentiment_model.zip"
extract_dir= r"~\BERT\bert_sentiment_model"

if not os.path.exists(extract_dir):
    with zipfile.Zipfile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

# 1. Load tokenizer and model
tokenizer= DistilBertTokenizer.from_pretrained(extract_dir)
model= TFDistilBertForSequenceClassification.from_pretrained(extract_dir)

# 2. Set Streamlit layout and background
st.set_page_config(page_title= 'Sentiment Classifier', layout= 'centered')

bg_image_url= 'https://media.bazaarvoice.com/Shutterstock_2247447401.png'
custom_css= f'''
<style>
    .stApp {{
        background-image: url("{bg_image_url}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }}
</style>
'''
st.markdown(custom_css, unsafe_allow_html= True)

st.title("BERT Sentiment Classifier")

# 3. User Input
user_input= st.text_area("Enter text to classify:", height= 150)

# 4. Classify button
if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Tokenize input
        inputs= tokenizer(
            user_input,
            truncation= True,
            padding= True,
            max_length= 128,
            return_tensors= "tf")
        
        # Predict
        logits= model(inputs).logits
        probs= tf.nn.softmax(logits, axis= 1).numpy()
        pred_class= np.argmax(probs)
        confidence= probs[0][pred_class]

        label_map= {0: 'Negative', 1: 'Positive'}

        # 5. DIsplay results
        st.markdown(f'### Prediction: **{label_map[pred_class]}')
        st.markdown(f'#### Confidence: `{confidence:.2%}`')
        st.balloons()