import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tools import load_data, verify_token_exist, get_path

verify_token_exist()

data = load_data(os.path.join(get_path(),'example_data.csv'), index_col="StudentID")
data = data.drop(["FirstName", "FamilyName"], axis=1)
data['family_ed'] = data['Fedu'] + data['Medu']


# Add a title to the app
st.title('Student Performance Analysis Summary')
st.info(" !!!!! Work to do: \
          better define features to the user or add legend. \
         In some case use more suited chart \
         (stack bar with cat. grade in horiz.axis or plotly distplot) !!!")

# Distribution of Final Grades
st.header('Distribution of Final Grades')
fig, ax = plt.subplots(figsize=(10, 8))
sns.countplot(data=data, x='FinalGrade', stat="count", ax=ax)
ax.set_title('Distribution of Final Grades of Students')
ax.set_xlabel('Final Grade')
ax.set_ylabel('Count')
st.pyplot(fig)

feats =[
['age', 'failures','family_ed', 'paid', 
 'Impact of failures and external help',],
['higher', 'reason', 'health', 'absences', 'Impact of wish and school attendance property'],
['Pstatus', 'guardian', 'famsup', 'famrel', 'Impact of some family property'],
['studytime', 'traveltime', 'activities', 'internet', 'Impact of time management'],
['romantic', 'freetime', 'goout', 'Dalc', 'Impact of social activities'],
]
# Influence of Failures, Family Education, and Desire for Higher Education
for f in feats:
    st.header(f[4])
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    sns.swarmplot(x=f[0], y='FinalGrade', data=data, ax=axs[0, 0])
    axs[0, 0].set_title(f'Impact of {f[0]} on Final Grade')
    axs[0, 0].set_xlabel(f'{f[0]}')
    axs[0, 0].set_ylabel('Final Grade')

    sns.swarmplot(x=f[1], y='FinalGrade', data=data, ax=axs[0, 1])
    axs[0, 1].set_title(f'Impact of {f[1]} on Final Grade')
    axs[0, 1].set_xlabel(f'{f[1]}')
    axs[0, 1].set_ylabel('Final Grade')

    sns.swarmplot(x=f[2], y='FinalGrade', data=data, ax=axs[1, 0])
    axs[1, 0].set_title(f'Impact of {f[2]} on Final Grade')
    axs[1, 0].set_xlabel(f'{f[2]}')
    axs[1, 0].set_ylabel('Final Grade')

    sns.swarmplot(x=f[3], y='FinalGrade', data=data, ax=axs[1,1])
    axs[1, 1].set_title(f'Impact of {f[3]} on Final Grade')
    axs[1, 1].set_xlabel(f'{f[3]}')
    axs[1, 1].set_ylabel('Final Grade')

    plt.tight_layout()
    st.pyplot(fig)


