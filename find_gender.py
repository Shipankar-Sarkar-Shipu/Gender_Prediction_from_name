import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle
import sys
clf= MultinomialNB()
dclf = DecisionTreeClassifier()
cv=CountVectorizer()

csvname=sys.argv[1]

print("\nFile Name: ", csvname)

filename = 'gender_pred_model.model'
filename_dt = 'gender_pred_model_dt.model'
vec_file="vectorizer.pickle"

def lower_case(s):
    return s.lower()
print(lower_case("kashfASF Hf ahfF"))



loaded_vectorizer = pickle.load(open(vec_file, 'rb'))
loaded_model_dt = pickle.load(open(filename_dt, 'rb'))
loaded_model = pickle.load(open(filename, 'rb'))

#Test
sample_name = ["Pasha"]
vect = loaded_vectorizer.transform(sample_name).toarray()
print("Pasha",loaded_model.predict(vect))

customer_df=pd.read_csv(f"{csvname}.csv")
x_test_df_pre= customer_df[['customer_id','name']].dropna()
x_test_df=x_test_df_pre['name']
X_test = loaded_vectorizer.transform(x_test_df).toarray()

results=loaded_model.predict(X_test)
results_dt=loaded_model_dt.predict(X_test)

result_df=pd.concat([x_test_df_pre, pd.Series(results)], axis=1).reset_index()
result_df=result_df.drop(["index"],axis=1)
result_df.rename(columns = {'name':'Name', 0:'Gender'}, inplace = True)
result_df['Gender'].replace({1:"Male",0:"Female"},inplace=True)
result_df.to_csv(f"Employee_Gender_{csvname}.csv")


result_df_dt=pd.concat([x_test_df_pre, pd.Series(results_dt)], axis=1).reset_index()
result_df_dt=result_df_dt.drop(["index"],axis=1)
result_df_dt.rename(columns = {'name':'Name', 0:'Gender'}, inplace = True)
result_df_dt['Gender'].replace({1:"Male",0:"Female"},inplace=True)
result_df_dt.to_csv(f"Employee_Gender_{csvname}_dt.csv")

print("."*10,"DONE","."*10)