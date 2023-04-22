import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import time

#Die Daten von der csv-Datei werden in die Variable data eingelesen
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/TravelInsurancePrediction.csv")

#Ausgabe der ersten sieben Columns des Datensets
print(data.head(7))

#Veränderung der Variable AnnualIncome --> Durch 10 dividieren --> zu hoch
data["AnnualIncome"] = data["AnnualIncome"].divide(10)

#Löschen der unrelevanten Daten
data.drop(columns=["Unnamed: 0"], inplace=True)

#Ausgabe aller summierten Nullwerte --> Ziel: Alles 0 --> Keine Nullwerte vorhanden im Datenset
print(data.isnull().sum())

#Minimales Alter in dem Datenset
print('Maximales Alter in dem Datenset: ' + str(data["Age"].min()))

#Maximales Alter in dem Datenset
print('Maximales Alter in dem Datenset: ' + str(data["Age"].max()))

#Ausgabe von Grundinformationen des Datensets
data.info()

#Veränderung der Variable TravelInsurance --> 0 wird zu Besitzt keine Reiseversicherung, 1 wird zu Besitzt eine Reiseversicherung
data["TravelInsurance"] = data["TravelInsurance"].map({0: "Besitzt keine Reiseversicherung", 1: "Besitzt eine Reiseversicherung"})

#Veränderung der Variable GraduateOrNot --> 0 wird zu No, 1 wird zu Yes
data["GraduateOrNot"] = data["GraduateOrNot"].map({"No": 0, "Yes": 1})

#Veränderung der Variable FrequentFlyer --> 0 wird zu No, 1 wird zu Yes
data["FrequentFlyer"] = data["FrequentFlyer"].map({"No": 0, "Yes": 1})

#Veränderung der Variable EverTravelledAbroad --> 0 wird zu No, 1 wird zu Yes
data["EverTravelledAbroad"] = data["EverTravelledAbroad"].map({"No": 0, "Yes": 1})


#????????????Veränderte Daten werden abgespeichert (könnte sonst zu Fehlern führen)???????????
data = data

#Visualisierung anhand der Variablen Age --> Altersverteilung
figure = px.histogram(data, 
                      x = "Age", 
                      color = "TravelInsurance", 
                      title= "Altersabhängigkeit bei dem Kauf einer Reiseversicherung")
figure.show()

#Visualisierung anhand der Variablen Employment Type
figure = px.histogram(data, 
                      x = "Employment Type", 
                      color = "TravelInsurance", 
                      title= "Beschäftigungsverhältnis bei dem Kauf einer Reiseversicherung")
figure.show()

#Visualisierung anhand der Variablen AnnualIncome
figure = px.histogram(data, 
                      x = "AnnualIncome", 
                      color = "TravelInsurance", 
                      title= "Einkommen in Abhängigkeit zum Kauf einer Reiseversicherung")
figure.show()

#In die Variable x werden alle Daten des Datensets bis auf die zu vorhersagende Zielvariable y (TravelInsurance) hinzugefügt. Vorhersage von y basiert auf x.
x = np.array(data[["Age", "GraduateOrNot", 
                   "AnnualIncome", "FamilyMembers", 
                   "ChronicDiseases", "FrequentFlyer", 
                   "EverTravelledAbroad"]])

#In die Variable y wird die zu vorhersagende Zielvariable TravelInsurance hinzugefügt
y = np.array(data[["TravelInsurance"]])

#Aufteilung von Training- und Testdaten
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20, random_state=42)

#Deklaration und Initialisierung von DecisionTree Algorithmus
model = DecisionTreeClassifier()

#Deklaration und Initialisierung von RandomForest Algorithmus
model2 = RandomForestClassifier()

#######################################################################
###########################DECISION TREE###############################
#######################################################################

#Laufzeitmessung Start DecisionTree Algorithmus
zeitanfangDecisionTree = time.time()

#Training des DecisionTree Algorithmus mit den Trainingsdaten
model.fit(xtrain, ytrain)

#Anhand der Testdaten können mithilfe des trainierten Algorithmus nun auch Vorhersagen getroffen werden
#predictions = model.predict(xtest)

#Wie viel kann der trainierte Algorithmus richtig vorhersagen?
print('Decision Tree Score: ' + str(model.score(xtest, ytest)))

#Laufzeitmessung Ende DecisionTree Algorithmus
zeitendeDecisionTree = time.time()

#Laufzeitmessung Berechnung und Ausgabe DecisionTree Algorithmus
print('Laufzeit für den Decision Tree: ' + str(zeitendeDecisionTree-zeitanfangDecisionTree))

#######################################################################
###########################RANDOM FOREST###############################
#######################################################################

#Laufzeitmessung Anfang RandomForest Algorithmus
zeitanfangRandomForest = time.time()

#Training des RandomForest Algorithmus mit den Trainingsdaten
model2.fit(xtrain, ytrain.ravel())

#Anhand der Testdaten können mithilfe des trainierten Algorithmus nun auch Vorhersagen getroffen werden
#predictions2 = model2.predict(xtest)

#Wie viel kann der trainierte Algorithmus richtig vorhersagen?
print('Random Forest Score: ' + str(model2.score(xtest, ytest)))

#Laufzeitmessung Ende RandomForest Algorithmus
zeitendeRandomForest = time.time()

#Laufzeitmessung Berechnung und Ausgabe RandomForest Algorithmus
print('Laufzeit für den Random Forest: ' + str(zeitendeRandomForest-zeitanfangRandomForest))


#TODO Usereingabe
userInputAge = int(input("Enter your Age: "))
#userInputEmploymentType = input("Enter your employment type (Private Sector/Self Employed OR Government Sector): ")
userInputGraduateOrNot = input("Graduated? Yes or No: ")
userInputAnnualIncome = float(input("AnnualIncome? Auf 5000 Euro genau gerundet: "))
userInputFamilyMembers = int(input("Amount of family members:  "))
userInputChronicDiseases = int(input("Chronic Diseases? Amount of chronic diseases: "))
userInputFrequentFlyer = input("Frequent Flyer? Yes or No: ")
userInputEverTravelledAbroad = input("Ever Travelled Abroad? Yes or No: ")

xpredict = np.array([userInputAge, userInputGraduateOrNot, userInputAnnualIncome, userInputFamilyMembers, userInputChronicDiseases, userInputFrequentFlyer, userInputEverTravelledAbroad]).reshape(1, -1)

print(xpredict)

print(model2.predict(xpredict))