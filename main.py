import streamlit as st
import pandas as pd
import numpy as np
import joblib
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import textblob
from textblob import TextBlob
import cleantext

lemmatizer=WordNetLemmatizer()

nltk.download("all")


# load the model from the file
rf=joblib.load("models/rf_sent_model.pkl")
count=joblib.load("models/count_vectorizer.pkl")

def preprocess_data(data):
    corpus=[]
    for i in data:
        mess=re.sub("[^a-zA-Z0-9]"," ",i)
        mess=mess.lower().split()
        mess=[lemmatizer.lemmatize(word) for word in mess if word not in stopwords.words("english")]
        mess=" ".join(mess)
        corpus.append(mess)
    return corpus    



# navigation bar
st.sidebar.image("images/nlp.jpg",width=500)
st.sidebar.title("Feedback Analyzer \n \n Developed by :  Gaurav Shrivastav & Mayur Mamtani ")


#file uploader 
upl=st.sidebar.file_uploader("Upload CSV File here",type=["csv"],accept_multiple_files=False)

# submit button
sub_btn=st.sidebar.button("Submit")


#main form
st.write("NLP Powered Feedback Analyzer")
txt=st.text_input("Enter your feedback here")

predict_btn=st.button("Predict")
clear_btn=st.button("Clear")


pre=st.text_input('Clean Text:')
if pre:
    st.write(cleantext.clean(pre,clean_all=False,extra_spaces=True, stopwords=True, lowercase=True, numbers=True, punct=True))


# function to create temporary file from their binary objects
def create_file(file_bytes,file_name):

    with open("tmp_files/"+file_name, 'wb') as f:
        f.write(file_bytes)

# if sub_btn:
#     #name of uploaded file
#     file_name=None
#     try:
#         file_name=file.name
#     except:
#         pass
#     if file_name:
#         file_bytes=file.getvalue()
#        # creating temorary files for resumes and jd from their respective binary objects
#         create_file(file_bytes,file_name)

#         df=pd.read_csv("tmp_files/"+file_name)
#         X=df["Feedback"]
#         X=preprocess_data(X)
#         count_x=count.transform(X)
#         y_pred=rf.predict(count_x)
#         y_pred=pd.DataFrame(y_pred)
#         y_pred.columns=["Predicted"]
#         df["Predicted"]=y_pred["Predicted"]
#         df.drop(columns=["Sentiment"],inplace=True,axis=1)
#         df.to_csv("outputs/"+file_name,index=False)
#         csv=df.to_csv().encode('utf-8')
#         st.download_button(
#             label="Download Results",
#             data=csv,
#             file_name='results.csv',
#             mime='text/csv',
#  )
#         st.success("Output file is saved in outputs folder")


       
#     else:
#         # warning message if no file is uploaded
#         st.warning("Please upload a csv file")

def score(x):
    blob1=TextBlob(x)
    return blob1.sentiment.polarity


def analyze(x):
    if x>=0.5:
        return 'Positive'
    elif x<= -0.20:
        return 'Negative'
    else:
        return 'Neutral'



if upl:
    df=pd.read_csv(upl)
    df['score']=df['Feedback'].apply(score)
    df['analysis']=df['score'].apply(analyze)
    st.write(df.head(10))

 
if predict_btn:
    if txt:
        X=[txt]
        X=preprocess_data(X)
        count_x=count.transform(X)
        y_pred=rf.predict(count_x)
        
        image_name=None
        if y_pred[0]==0:
            image_name="images/neg.jpg"
        else:
            image_name="images/pos.jpg"

        result_image=st.image(image=image_name,width=500)

        blob=TextBlob(txt)
        st.write('Polarity:', round(blob.sentiment.polarity,2))
        st.write('Subjectivity:',round(blob.sentiment.subjectivity,2))
    else:
        st.warning("Please Enter a Feedback to Predict")


if clear_btn:
    result_image=None
