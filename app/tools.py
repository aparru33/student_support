import streamlit as st
import messages as msg
import pandas as pd

def verify_token_exist():
    if "token" not in st.session_state:
        st.error(msg.NOT_AUTH)
        st.stop()

@st.cache_data
def load_data(file_path,sep = ';', encoding = 'utf-8', index_col=False):
    return pd.read_csv(file_path, sep=sep, encoding=encoding, index_col=index_col)

def prepare_data(data):
    model_data = data.drop(['FirstName', 'FamilyName','PredictedGrade', 'SchoolID', 'StudentID'], axis=1)
    model_data['sex' ]=model_data['sex'].map({'M':0 ,'F':1})
    model_data['address' ]=model_data['address'].map({'R':0 ,'U':1})
    model_data['famsize' ]=model_data['famsize'].map({'LE3':0 ,'GT3':1})
    model_data['Pstatus' ]=model_data['Pstatus'].map({'A':0 ,'T':1})
    model_data['famsup' ]=model_data['famsup'].map({'no':0, 'yes':1})
    model_data['schoolsup' ]=model_data['schoolsup'].map({'no':0, 'yes':1})
    model_data['paid' ]=model_data['paid'].map({'no':0, 'yes':1})
    model_data['activities' ]=model_data['activities'].map({'no':0, 'yes':1})
    model_data['nursery' ]=model_data['nursery'].map({'no':0, 'yes':1})
    model_data['higher' ]=model_data['higher'].map({'no':0, 'yes':1})
    model_data['internet' ]=model_data['internet'].map({'no':0, 'yes':1})
    model_data['romantic' ]=model_data['romantic'].map({'no':0, 'yes':1})
    for p in ['M','F']:
        for c in ['health', 'other', 'services', 'teacher']:
            model_data[p+'job_'+c] = model_data[p+'job'].map(lambda x :1 if x==c else 0)
    for c in ['home', 'other', 'reputation',]:
            model_data['reason_'+c] =model_data['reason'].map(lambda x :1 if x==c else 0)
    for c in ['mother', 'other']:
            model_data['guardian_'+c] = model_data['guardian'].map(lambda x :1 if x==c else 0)
    return model_data.drop(['Fjob','Mjob','guardian','reason'], axis=1)
