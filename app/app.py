import streamlit as st
from messages import ERROR_LOGIN
from tools import load_data, get_path
import os
# Load CSV files as dataframes

users = load_data(os.path.join(get_path(),'cunselors.csv'))
schools = load_data(os.path.join(get_path(),'schools.csv'))
students = load_data(os.path.join(get_path(),'data/students.csv'))

def login(userName:str, password:str):
    """
    Check if a row of a DataFrame has both the value of userName
    and the value of password"
    
    Args:
    - df: DataFrame
    
    Returns:
    - result: Boolean indicating if credentials are present in any row
    """
    print(users['userName'])
    user=users.loc[(users['userName'] == userName) & (users['password'] == password)]
    count = user.shape[0]
    if count!=1:
        return {"status": "error", "error": {"message":ERROR_LOGIN}}
    st.session_state["user"] = user
    return {"status": "ok", "response": {"token":userName.__hash__}}

st.title("Welcome to Help Student")
 
if "token" not in st.session_state:
    with st.form(key="login_form"):
        st.info('user test : try user a with password pwd1')
        _user_euserName = st.text_input("Enter your user name")
        _user_password = st.text_input("Enter your password", type="password")
        submit_button = st.form_submit_button(label="Login")

        if submit_button:  # This checks if the button is pressed.
            result = login(_user_euserName, _user_password)
            print(result)
            if result["status"] == "ok":
                st.success("Login Successful")
                st.session_state["token"] = result["response"]["token"]
                st.session_state["students"] = students
            else:
                st.error(result["error"]["message"])
else:
    st.write("Already logged in, enjoy the app!")
