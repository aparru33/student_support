import streamlit as st
import pandas as pd
import joblib
from tools import verify_token_exist, load_data, prepare_data
from messages import GOOD_PREDICT, BAD_PREDICT
verify_token_exist()


# Load the existing student data from the CSV file
data_path = 'data/students.csv'
students_df = load_data(data_path)
# Function to predict grades and store student data
def predict_grade(student_data):
    print('----------- predict_grade() ---------')
    print(student_data)
    # Load the trained model (you can replace this with your actual model loading code)
    model = joblib.load('model/model.joblib')

    # Predict the grade for the new student data
    prepared_data =prepare_data(student_data)
    print(prepared_data)
    predicted_grade = model.predict(prepare_data(student_data))
    print(predicted_grade)
    # Add the predicted grade to the student data
    student_data['PredictedGrade'] = predicted_grade[0]
    # Append the student data to the existing DataFrame and save to CSV
    updated_df = pd.concat([students_df, student_data], ignore_index=True)
    updated_df.to_csv(data_path, index=False)
    return predicted_grade[0]


edu= ['none','primary education (4th grade)','5th to 9th grade','secondary education', 'higher education']

travel_time = ['<15 min','15 to 30 min', '30 min. to 1 hour', '>1 hour']

study_time = ['1 - <2 hours','2 to 5 hours','5 to 10 hours','>10 hours']
low_high = ['Very low', 'Low', 'Medium', 'High', 'Very high']
bad_good = ['Very bad', 'Bad', 'Neutral', 'Good', 'Very good']

def add_student():
    # Add new student data
    st.header('Add New Student')

    col1, col2, col3 = st.columns(3)
    with col1:
        first_name = st.text_input('First Name')
        age = st.number_input('Age', min_value=10, max_value=25)
        guardian = st.selectbox('Guardian', ['mother', 'father', 'other'])
        reason = st.selectbox('Reason for Choosing School', ['home', 'reputation', 'course', 'other'])
        paid = st.selectbox('Extra Paid Classes', ['yes', 'no'])
        activities = st.selectbox('Extra-Curricular Activities', ['yes', 'no'])
        internet = st.selectbox('Internet Access', ['yes', 'no'])
        pstatus = st.selectbox('Parent Cohabitation Status', ['Living together', 'Apart'])
        famrel = st.selectbox('Family Relationship quality', bad_good)
        dalc = st.selectbox('Workday Alcohol Consumption', low_high)
        absences = st.number_input('Absences', min_value=0)

    with col2:
        family_name = st.text_input('Family Name')
        medu = st.selectbox('Mother Education', edu)
        mjob = st.selectbox('Mother Job', ['at_home', 'teacher', 'health', 'services', 'other'])
        traveltime = st.selectbox('Travel Time', travel_time)
        schoolsup = st.selectbox('School Support', ['yes', 'no'])
        nursery = st.selectbox('Attended Nursery', ['yes', 'no'])
        address = st.selectbox('Address', ['Urban', 'Rural'])
        freetime = st.selectbox('Free Time', low_high)
        walc = st.selectbox('Weekend Alcohol Consumption', low_high)
        failures = st.number_input('failures', min_value=0)

    with col3:
        sex = st.selectbox('Sex', ['M', 'F'])
        famsize = st.selectbox('Family Size', ['GT3', 'LE3'])
        fedu = st.selectbox('Father Education', edu)
        fjob = st.selectbox('Father Job', ['at_home', 'teacher', 'health', 'services', 'other'])
        studytime = st.selectbox('Study Time', study_time)
        famsup = st.selectbox('Family Support', ['yes', 'no'])
        higher = st.selectbox('Wants Higher Education', ['yes', 'no'])
        romantic = st.selectbox('Romantic Relationship', ['yes', 'no'])
        goout = st.selectbox('Going Out', low_high)
        health = st.selectbox('Health Status', bad_good)


    # Predict and store grade for new student
    if st.button('Add Student'):
        
        user = st.session_state["user"]
        new_student_data = {
            'StudentID': [max(students_df.StudentID)+1],
            'FirstName': [first_name],
            'FamilyName': [family_name],
            'sex': [sex],
            'age': [age],
            'address': [address[0]],  # Assuming address is a string
            'famsize': [famsize],
            'Pstatus': ['T' if pstatus[0]=='L' else 'A'],  # Assuming pstatus is a string
            'Medu': [edu.index(medu)],
            'Fedu': [edu.index(fedu)],
            'Mjob': [mjob],
            'Fjob': [fjob],
            'reason': [reason],
            'guardian': [guardian],
            'traveltime': [travel_time.index(traveltime)+1],
            'studytime': [study_time.index(studytime)+1],
            'failures': [failures],
            'schoolsup': [schoolsup],
            'famsup': [famsup],
            'paid': [paid],
            'activities': [activities],
            'nursery': [nursery],
            'higher': [higher],
            'internet': [internet],
            'romantic': [romantic],
            'famrel': [bad_good.index(famrel)+1],
            'freetime': [low_high.index(freetime)+1],
            'goout': [low_high.index(goout)+1],
            'Dalc': [low_high.index(dalc)+1],
            'Walc' :[low_high.index(walc)+1],
            'health': [bad_good.index(health)+1],
            'absences': [absences],
            'SchoolID' :user.loc[0].SchoolID,
            'PredictedGrade': [0]  # Initial value, to be updated after prediction
        }
        new_student_df = pd.DataFrame.from_dict(new_student_data)
        predicted_grade = min(round(predict_grade(new_student_df),2)*20,20)
        new_student_df.loc[0, 'PredictedGrade'] = predicted_grade
        new_student_df.to_csv(data_path, mode='a', header=False, index=False)
        st.success(f'Predicted Grade: {predicted_grade}. \
                        {BAD_PREDICT if predicted_grade<10 else GOOD_PREDICT }')
        
        students_df = load_data(data_path)    

# Streamlit UI
st.title('Student Performance Analysis and Prediction')
adding = False
consulting = False
but1, but2 = st.columns(2)
with but1:
    if(st.button('Add a student')):
        adding=True
        consulting=False
with but2:
    if st.button('Consult existing student'):
        adding=False
        consulting=True

if adding:
    add_student()
elif consulting:
    st.write('Work in progress')

st.write('idea: add one chatbot enhanced with RAG to give advice based of grade of the student')