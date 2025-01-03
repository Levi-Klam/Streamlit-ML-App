import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, PowerTransformer, MaxAbsScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import numpy as np
from numpy import ravel
import pandas as pd

#Loading Dataset
df = pd.read_csv("datasets/loan_data.csv")
#Setting features
features = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
]
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
df['Property_Area'] = df['Property_Area'].map({'Rural': 0, 'Urban': 1, 'Semiurban': 2})
df['Dependents'] = df['Dependents'].replace('3+', '3')

cat_imputer = SimpleImputer(strategy="most_frequent")
df[['Gender', 'Dependents', 'Self_Employed']] = cat_imputer.fit_transform(df[['Gender', 'Dependents', 'Self_Employed']])
num_imputer = SimpleImputer(strategy="median")
df['Loan_Amount_Term'] = num_imputer.fit_transform(df[['Loan_Amount_Term']])
df['Credit_History'] = cat_imputer.fit_transform(df[['Credit_History']])

y = df['Loan_Status']
X = df[features]
X.head()

encoder = LabelEncoder()
y = encoder.fit_transform(ravel(y))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=1/3,
    random_state=0)

# print(X_train.shape)
# print(X_test.shape)

# Title
st.title(" Loan Status Predictor - CIS 335 Project")
#Classifier Box
classifier_name = st.sidebar.selectbox(label="Select Classifer", options=["Random Forest", "ADA Boost", "SVM", "Decision Tree"])
st.write(f'Current Classifer is {classifier_name} ')
normalization = st.sidebar.selectbox(label="Select Normalization", options=["None", "Z-Score", "Min-Max"])
st.write(f'Current Normalization type is {normalization}')

def get_normalization(df, normalization):
    if normalization == 'Z-Score':
        scaler = StandardScaler()
        df_normalized = df
        df_normalized[df_normalized.columns] = scaler.fit_transform(df_normalized)
        return df_normalized
    elif normalization == 'Min-Max':
        scaler = MinMaxScaler()
        df_normalized = df
        df_normalized[df_normalized.columns] = scaler.fit_transform(df_normalized)
        return df_normalized
    else:
        return df
    
#UI 
def add_parameter_ui(classifier_name):
    params = dict()
    if classifier_name == 'Random Forest':
        criterion = st.sidebar.selectbox(label="Select Criterion", options=["gini", "entropy", "log_loss"])
        st.write(f'Current Criterion is {criterion}')
        max_depth = st.sidebar.slider("max depth", 2, 10)
        st.write(f'Current Depth is {max_depth}')
        n_estimators = st.sidebar.slider("n_estimators", 2, 50)
        st.write(f'Current Estimator is {n_estimators}')
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
        params["criterion"] = criterion
        
    elif classifier_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        st.write(f'Current C is {C}')
        params["C"] = C
        degree = st.sidebar.selectbox(label="Degree", options=[2, 3, 4, 5])
        st.write(f'Current Degree is {degree}')
        params["Degree"] = degree
        iterations = st.sidebar.selectbox(label="Max Iterations (-1 = default, no limit)", options=[-1, 10, 50, 100, 500, 1000])
        st.write(f'Current Iterations is {iterations}')
        params["Max Iteration"] = iterations
        
    elif classifier_name == "ADA Boost":
        # n_classifiers = st.sidebar.selectbox(label="Number of Classifiers", options=[10, 25, 50, 100, 200, 500])
        n_classifiers = st.sidebar.slider("Number of Classifiers", 2, 250)
        learn_rate = st.sidebar.selectbox(label="Learning Rate", options=[.1, 1, 2, 10])
        st.write(f'Current learning rate is {learn_rate}')
        st.write(f'Current number of classifiers is {n_classifiers}')
        params["Learn Rate"] = learn_rate
        params["N Classifiers"] = n_classifiers

    elif classifier_name == 'Decision Tree':
        criterion = st.sidebar.selectbox(label="Select Criterion", options=["gini", "entropy"])
        st.write(f'Current criterion is {criterion}')
        params["criterion"] = criterion
        max_depth = st.sidebar.selectbox(label="max depth", options=[2, 4, 6, 8, 10])
        st.write(f'Current depth is {max_depth}')
        params["max_depth"] = max_depth
        
    return params
            
def get_classifier(classifier_name, parameters):
    if classifier_name == "Random Forest":
        clf = RandomForestClassifier(criterion=parameters["criterion"],
                                     max_depth=parameters["max_depth"],
                                     n_estimators=parameters["n_estimators"])
    elif classifier_name == 'Decision Tree':
        clf = DecisionTreeClassifier(max_depth=parameters['max_depth'])
    elif classifier_name == "SVM":
        clf = SVC(C=params["C"], degree=params["Degree"], max_iter=params["Max Iteration"])
    elif classifier_name == "ADA Boost":
        clf = AdaBoostClassifier(n_estimators=params["N Classifiers"], learning_rate=params["Learn Rate"])  
    return clf  

params = add_parameter_ui(classifier_name)
classifier = get_classifier(classifier_name, params)

classifier.fit(get_normalization(X_train, normalization), y_train)
predictions = classifier.predict(get_normalization(X_test, normalization))
def get_score(y_test, predictions):
    score = accuracy_score(y_test, predictions)
    return score
st.write(f'Score is : ' + str(get_score(y_test, predictions)))
st.write(f'Shape of dataset is : {X.shape}')
st.write(f'number of classes in the dataset is : {len(np.unique(y))}')
