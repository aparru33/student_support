#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
#%%
data = pd.read_csv("data/exercice_data.csv", sep=",", encoding = "cp1252", index_col='StudentID' )

#%%
## Target
# Create a model to see student to help
# Create a dashboard to allow teachers to visualize data depending on complexity and value of help
# One graph with grade on one axe and help complexity on a second

## Criteria
# Visualization and presented data are invaluable for users.
# Ease of deployment is crucial.
# Simplicity in maintenance and enhancement is key.
# Relevance of the ideas is paramount.

# -------------------------------------------------------------------------------------------------
## What is a student in difficulty:
# Mainly a student at risk for receiving less than "pass" i.e less than 10 in ou the grading system
# An other definition could be student performing less than usual but we haven't the grade per time
# so we can't use this idea

## What help the app can give to the teacher:
# Firt, detect the student before a critical acedemical incident occurred
# Second, gives advice depending of the situation of the student to define a management plan 
# (maybe use a LLM with RAG using adademic, clinical, sociological, psychological corpus)
# Use “K-Salts” (knowledge, skills, attitude, learner, teacher, system) to display diagnosis

## Data analysis
# Not not so many time available so I pass this part although I will do some on the streamlit app
# I will use feature selection to choose relevant feature
# and then use them to show student having bad data in them

## What kind of problem do we have
# The problem is supervised since we have the final grade. 
# It could be 
# - binary classification meaning final grade less or greater than 10 (less -> à, greater -> 1)
# - 5-level classification ( we could use ) or 6 level classification
# (https://en.wikipedia.org/wiki/Academic_grading_in_Portugal)
# - regression on final grade

# As the tool is here to prioritize students to help we keep two approach:
# -  6 level classification because the data are from secondary schooland weeak and poor would be 
# the student to priorize with poor first and weak next
# - regression on final grade 

# %%
# data cleaning
for k,v in dict(data.isna().sum()).items():
    if v>0: print(k, 'has', v,'missing values')
# the data is already clean
#%%
# for c in model_data.columns:
#     print(c, model_data[c].unique() )

#%%
def prepare_data(data):
    model_data = data.drop(['FirstName', 'FamilyName'], axis=1)
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
    
    #we use one hot encoding for these features
    # as there isn't ordinality between the values and no binary category
    model_data = pd.get_dummies(model_data, 
                        prefix = ["Mjob"], columns = ["Mjob"], dtype=int, drop_first=True)
    model_data = pd.get_dummies(model_data,
                        prefix = ["Fjob"], columns = ["Fjob"], dtype=int, drop_first=True)
    model_data = pd.get_dummies(model_data, 
                        prefix = ["reason"], columns = ["reason"], dtype=int, drop_first=True)
    model_data = pd.get_dummies(model_data, 
                        prefix = ["guardian"], columns = ["guardian"], dtype=int, drop_first=True)
    return model_data

model_data = prepare_data(data)
print(model_data.columns)
#%%
# corr = model_data.corr()
# corr.style.background_gradient(cmap='coolwarm')

# %%
# create train and test set
from sklearn.model_selection import train_test_split
df_train, df_validation = train_test_split(model_data.copy(),test_size=0.15)

Y_reg = df_train.pop("FinalGrade")
Y_reg=Y_reg/20


Y_reg_val = df_validation.pop("FinalGrade")
Y_reg_val=Y_reg_val/20

# some of the case in comment are missing in the dataset
# so we transform the problem in a binary classification
def grade_2_cat(x):
    # if x<=3.4:
    #     return 0
    if x<0.5:
        return 0
    else:
        return 1
    # elif x<= 15.4:
    #     return 3
    # elif x<= 17.4:
    #     return 4
    # elif x<= 17.4:
    #     return 5
    #else: return 3#6 some of the case in comment are missing in the dataset

Y_cat = Y_reg.map(grade_2_cat)
X = df_train

Y_cat_val = Y_reg_val.map(grade_2_cat)
X_val = df_validation

#%%
# model selection
# Define classification models to evaluate

# model to test:
# classification : random forest classifier, gradient boosting classifier, SVM classifier,
# multinomial logistic regression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import precision_score, accuracy_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import sklearn
import xgboost as xgb

def get_pred_acc(_model,X_test,y_test):
    y_pred = _model.predict(X_test)

    # Calculate precision
    precision = precision_score(y_test, y_pred)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    recall = recall_score(y_test, y_pred)

    #print(f'Precision: {precision:.4f}')
    #print(f'Accuracy: {accuracy:.4f}')
    return precision, accuracy, recall

def get_confusion_matrix(_model,X_test,y_test):
    y_pred = _model.predict(X_test)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(3, 2))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    

#%%
overall_results = {}
results = {}

# Regression works better so we don't keep classifier model. 
# Also regression could help visualization
# models = {
#     'Logistic Regression classifier': LogisticRegression(max_iter=1000),
#     'Random Forest classifier': RandomForestClassifier(),
#     'xgboost classifier' : xgb.XGBClassifier(objective='binary:logistic',tree_method="hist")
# }

# # Define cross-validation strategy (Stratified K-Fold) (k=4 is optimal, find with a loop)
# s = 4
# res = f"\nn_split = {s}"
# cv = StratifiedKFold(n_splits=s, shuffle=True, random_state=42)

# # Perform model selection and evaluation using cross-validation
# for name, model in models.items():
#     scores = cross_val_score(model, X, Y_cat, cv=cv, scoring='roc_auc')
#     avg_prec, avg_acc, avg_rec = 0, 0, 0
#     n = 0
    
#     for train_index, val_index in cv.split(X, Y_cat):
#         # Split the data into training and validation sets
#         X_train, X_val = X.iloc[train_index], X.iloc[val_index]
#         Y_train, Y_val = Y_cat.iloc[train_index], Y_cat.iloc[val_index]

#         m = model.fit(X_train, Y_train)
#         prec, acc, rec = get_pred_acc(m, X_val, Y_val)
#         avg_prec += prec
#         avg_acc += acc
#         avg_rec+= rec
#         n += 1

#     avg_prec /= n  # Average precision across folds
#     avg_acc /= n   # Average accuracy across folds
#     avg_rec /= n 
#     results[name] = [np.mean(scores), avg_prec, avg_acc, avg_rec, model]

# # Display results
# for name, scores in results.items():
#     #print(f'{name}: Mean AUC = {np.mean(scores[0]):.4f}, Std = {np.std(scores[0]):.4f},' + \
#     #    f' precision = {scores[1]:.4f}, accuracy = {scores[2]:.4f}')
    
#     res+='\n'+f'{name}: Mean AUC = {np.mean(scores[0]):.4f}, Std = {np.std(scores[0]):.4f},' + \
#         f' precision = {scores[1]:.4f}, accuracy = {scores[2]:.4f}'
#     overall_results[name] = [round(scores[1],4), round(scores[2],4), round(scores[3],4), scores[4]]

# print(res)
# %%
# Define regression models to evaluate
# Keep in mind that we don't have enough data to properly choose the most efficient model
# Some model like k-nn could be better with so few data but would not scale well on more data
# regression: linear regression, random forest, xgboost, SVR (for NN we don't have enough data,
# we could do data augmentation though)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error


# Define classification models to evaluate
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regression': DecisionTreeRegressor(),
    'Random ForestRegression': RandomForestRegressor(),
    'Extra Tree Regression':ExtraTreesRegressor(),
    'SVR Regression' : SVR(),
    'GradientBoostingRegressor' : GradientBoostingRegressor(),
    'xgboost Regression' : xgb.XGBRegressor(objective='binary:logistic',tree_method="hist")
}
#%%
from sklearn.model_selection import KFold

results = {}
kf = KFold(n_splits=4, shuffle=True, random_state=42)
for name, model in models.items():
    scores = -cross_val_score(model, X, Y_reg, cv=kf, scoring='neg_mean_absolute_error')  # Convert negative MAE back to positive
    
    avg_prec, avg_acc, avg_rec =0,0,0
    n=0
    for train_index, val_index in kf.split(X):
        # Split the data into training and validation sets
        X_train, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        Y_train, Y_val_fold = Y_reg.iloc[train_index], Y_reg.iloc[val_index]

    #for i in range(nb):
        m=model.fit(X_train, Y_train)
        Y_pred = m.predict(X_val_fold)
        Y_pred2cat = [grade_2_cat(x) for x in Y_pred]
        Y_val_fold = Y_val_fold.map(grade_2_cat)
        avg_prec += precision_score(Y_val_fold, Y_pred2cat)
        # Calculate accuracy
        avg_acc += accuracy_score(Y_val_fold, Y_pred2cat)
        avg_rec += recall_score(Y_val_fold, Y_pred2cat)
        n+=1
    results[name] = [np.mean(scores), avg_prec/n ,avg_acc/n,avg_rec/n, model]
# Display results
#%%
dash_str = '-----------------------------------------------------------'
for name, scores in results.items():
    print(dash_str)
    print(f'{name}')
    print(f'CV Mean MAE = {scores[0]:.4f} precision = {scores[1]:.4f}, accuracy = {scores[2]:.4f}, recall = {scores[3]:.4f} ')
    
    overall_results[name] = [round(scores[1],4), round(scores[2],4), round(scores[3],4), scores[4]]
    #overall_results[name] = f' precision = {scores[1]:.4f}, accuracy = {scores[2]:.4f}'
# %%
#all results
best_prec = ('', 0, 0, 0, None)
best_acc = ('', 0, 0, 0, None)
best_recall = ('', 0, 0, 0, None)

for model, metrics in overall_results.items():
    precision, accuracy, recall, _ = metrics
    print(f'{model}: Precision = {precision}, Accuracy = {accuracy}, Recall = {recall}')
    
    # Update best precision
    if precision > best_prec[1]:
        best_prec = (model, precision, accuracy, recall, _)
    elif precision == best_prec[1] and accuracy > best_prec[2]:
        best_prec = (model, precision, accuracy, recall, _)

    # Update best accuracy
    if accuracy > best_acc[2]:
        best_acc = (model, accuracy, precision, recall, _)
    elif accuracy == best_acc[2] and precision > best_acc[1]:
        best_acc = (model, accuracy, precision, recall, _)

    # Update best recall
    if recall > best_recall[3]:
        best_recall = (model, recall, precision, accuracy, _)
    elif recall == best_recall[3] and precision > best_recall[1]:
        best_recall = (model, recall, precision, accuracy, _)

print(f"\nBest Precision: {best_prec[0]} (Precision={best_prec[1]}, Accuracy={best_prec[2]}, Recall={best_prec[3]})")
print(f"Best Accuracy: {best_acc[0]} (Accuracy={best_acc[1]}, Precision={best_acc[2]}, Recall={best_acc[3]})")
print(f"Best Recall: {best_recall[0]} (Recall={best_recall[1]}, Precision={best_recall[2]}, Accuracy={best_recall[3]})")
#%%
# we keep the model with the best precision. With so few data,
# the selected model could change on a new run
# If time available should do hypertuning
Y = model_data.pop("FinalGrade")/20
#%%
# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(model_data, Y, test_size=0.2, random_state=42)

# Fit the Lasso model on the training data
mm = best_prec[4]
model = mm.fit(X_train, Y_train.map(grade_2_cat))
test_set = X_test.copy()
# Predict the target variable on the testing data
Y_pred = model.predict(X_test)

# Convert the predicted continuous values to binary labels
Y_pred = [grade_2_cat(x) for x in Y_pred]
Y_test_cat=Y_test.map(grade_2_cat)

#%% export test set
# Combine X_test and predicted labels into a DataFrame
test_set = pd.concat([X_test, pd.Series(Y_pred, name='Predicted Grade', index=X_test.index)], axis=1)

# Export the test set to a CSV file
test_set.to_csv('data/test_set.csv', index=False)


#%%
# Calculate precision and accuracy
prec = precision_score(Y_test_cat, Y_pred)
acc = accuracy_score(Y_test_cat, Y_pred)
recall = recall_score(Y_test_cat, Y_pred)

print("Precision:", prec)
print("Accuracy:", acc)
print("Recall:", recall)
#%% Get the features importance. As for the choice of model, the scarcity of data
## doesn't allow to choose the feature for sure
if isinstance(model, LinearRegression) or isinstance(model, Ridge) or isinstance(model, Lasso):
    importances = model.coef_
else:
    importances = model.feature_importances_

# Get the feature names if available
if hasattr(model, 'feature_names_in_'):
    feature_names = model.feature_names_in_
else:
    # If feature names are not available, use generic names like f1, f2, ...
    n_features = X_train.shape[1]
    feature_names = [f'f{i}' for i in range(1, n_features + 1)]

# Create a dictionary to store feature importance
feature_importance = dict(zip(feature_names, importances))

# Sort the dictionary by absolute coefficient values
sorted_feature_importance = dict(sorted(feature_importance.items(), key=lambda item: abs(item[1]), reverse=True))
print(f"The model has {len(feature_importance)}")
# Print or visualize the sorted feature importance
n=0

print(dash_str)
print("10 most important features")
print(dash_str)
for feature, importance in sorted_feature_importance.items():
    
    print(f'{feature}: {importance}')
    if n==10:
        print(dash_str)
        print("Other features")
        print(dash_str)
    n+=1


#%%
# export the model
joblib.dump(model, f'{best_prec[0]}.joblib')

# %%
