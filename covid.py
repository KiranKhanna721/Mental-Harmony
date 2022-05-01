import pandas as pd
import numpy as np
import string
import re
import nltk
import streamlit as st
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords 
stopwords=stopwords.words('english')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
from sklearn.svm import LinearSVC
lsvc = LinearSVC(random_state = 2021)


def data_prep(text):
  text=text.lower()
  text = re.sub(r'[^\w\s]', '', text)
  text=" ".join(t for t in text.split() if t not in stopwords)
  return text

df = pd.read_csv('covid.csv')
df["questions"]=df["questions"].apply(data_prep)
tf=TfidfVectorizer()
tf_train=tf.fit_transform(df["questions"])
df_check=pd.DataFrame(tf_train.toarray(),columns=tf.get_feature_names())
df["Answers_Code"]=le.fit_transform(df["answers"])
lsvc.fit(df_check,df["Answers_Code"])
Ans=df["answers"].unique()
Ans=Ans.tolist()
Ans_Code=df["Answers_Code"].unique()
Ans_Code=Ans_Code.tolist()

def app():
    st.title("Covid 19 chatbot")
    st.image("https://s17776.pcdn.co/wp-content/uploads/2020/07/Healthcare-chatbot-sm.jpg")
    text = st.selectbox("Example questions related to mental healthcare",(st.text_input("Ask any question related to covid 19 "),"How does COVID-19 spread?"))
    if text!=None:
        tx =[]
        tx.append(text)
        testing=tf.transform(tx)
        result = lsvc.predict(testing)[0]
        r  = Ans_Code.index(result)
        if st.button('Submit'):
          st.write(Ans[r])