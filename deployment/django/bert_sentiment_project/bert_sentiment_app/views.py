import os
import numpy as np
from django.shortcuts import render
from django.http import HttpResponse
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf

# Load the model and toeknizer once
model_path= r'~\BERT\bert_sentiment_model'
tokenizer= DistilBertTokenizer.from_pretrained(model_path)
model= TFDistilBertForSequenceClassification.from_pretrained(model_path)

def predict_sentiment(text):
    encoding= tokenizer(
        text,
        truncation= True,
        padding= True,
        max_length= 128,
        return_tensors= 'tf'
    )
    outputs= model(encoding)
    logits= outputs.logits
    probs= tf.nn.softmax(logits, axis=1).numpy()[0]
    prediction= np.argmax(probs)
    confidence= probs[prediction] * 100

    label= "Positive" if prediction== 1 else "Negative"
    return label, round(confidence, 2)

def sentiment_view(request):
    result= None
    confidence= None
    input_text= ""

    if request.method== "POST":
        input_text= request.POST.get("text_input", "")
        #print("Received text:", input_text)  # DEBUGGING
        if input_text:
            result, confidence= predict_sentiment(input_text)
            #print("Prediction result:", result)  # DEBUGGING

    return render(request, "sentiment_form.html", {
        "prediction": result,
        "input_text": input_text,
        "confidence": confidence
    })
