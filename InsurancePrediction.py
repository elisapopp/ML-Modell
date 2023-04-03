import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/TravelInsurancePrediction.csv")
data.head()

data.drop(columns=["Unnamed: 0"], inplace=True)
data.isnull().sum()
data.info()

data["TravelInsurance"] = data["TravelInsurance"].map({0: "Besitzt keine Reiseversicherung", 1: "Besitzt eine Reiseversicherung"})

data = data

#Age
figure = px.histogram(data, x = "Age", 
                      color = "TravelInsurance", 
                      title= "Altersabhängigkeit bei dem Kauf einer Reiseversicherung")
figure.show()

#Employment Type
figure = px.histogram(data, x = "Employment Type", 
                      color = "TravelInsurance", 
                      title= "Beschäftigungsverhältnis bei dem Kauf einer Reiseversicherung")
figure.show()

#Annual Income
figure = px.histogram(data, x = "AnnualIncome", 
                      color = "TravelInsurance", 
                      title= "Einkommen in Abhängigkeit zum Kauf einer Reiseversicherung")
figure.show()


data["GraduateOrNot"] = data["GraduateOrNot"].map({"No": 0, "Yes": 1})
data["FrequentFlyer"] = data["FrequentFlyer"].map({"No": 0, "Yes": 1})
data["EverTravelledAbroad"] = data["EverTravelledAbroad"].map({"No": 0, "Yes": 1})
x = np.array(data[["Age", "GraduateOrNot", 
                   "AnnualIncome", "FamilyMembers", 
                   "ChronicDiseases", "FrequentFlyer", 
                   "EverTravelledAbroad"]])
y = np.array(data[["TravelInsurance"]])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
model = DecisionTreeClassifier()
model2 = RandomForestClassifier()
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)

#userInputAge = input("Enter your Age: ")

print('Decision Tree Score: ' + str(model.score(xtest, ytest)))

model2.fit(xtrain, ytrain.ravel())
predictions2 = model2.predict(xtest)
print('Random Forest Score: ' + str(model2.score(xtest, ytest)))