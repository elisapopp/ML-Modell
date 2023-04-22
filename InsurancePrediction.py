import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import time
import matplotlib.pyplot as plt

#Die Daten von der csv-Datei werden in die Variable data eingelesen
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/TravelInsurancePrediction.csv")

#Ausgabe der ersten sieben Einträge des Datensets
print("Ausgabe der ersten sieben Einträge des Datensets")
print(data.head(7))

#Veränderung der Variable AnnualIncome --> Durch 10 dividieren --> zu hoch, Daten höchstwahrscheinlich falsch
data["AnnualIncome"] = data["AnnualIncome"].divide(10)

#Löschen der unrelevanten Daten (erster Eintrag)
data.drop(columns=["Unnamed: 0"], inplace=True)

#Ausgabe aller summierten Nullwerte --> Ziel: Alles 0 --> Keine Nullwerte vorhanden im Datenset
print("=======================================================================================================")
print("Ausgabe aller summierten Nullwerte --> Ziel: Alles 0 --> Keine Nullwerte vorhanden im Datenset")
print(data.isnull().sum())

#Minimales Alter in dem Datenset
print("=======================================================================================================")
print('Maximales Alter in dem Datenset: ' + str(data["Age"].min()))

#Maximales Alter in dem Datenset
print("=======================================================================================================")
print('Maximales Alter in dem Datenset: ' + str(data["Age"].max()))

#Ausgabe von Grundinformationen des Datensets
print("=======================================================================================================")
data.info()

#Veränderung der Variable TravelInsurance --> 0 wird zu Besitzt keine Reiseversicherung, 1 wird zu Besitzt eine Reiseversicherung
data["TravelInsurance"] = data["TravelInsurance"].map({0: "Besitzt keine Reiseversicherung", 1: "Besitzt eine Reiseversicherung"})

#Veränderung der Variable GraduateOrNot --> 0 wird zu No, 1 wird zu Yes
data["GraduateOrNot"] = data["GraduateOrNot"].map({"No": 0, "Yes": 1})

#Veränderung der Variable FrequentFlyer --> 0 wird zu No, 1 wird zu Yes
data["FrequentFlyer"] = data["FrequentFlyer"].map({"No": 0, "Yes": 1})

#Veränderung der Variable EverTravelledAbroad --> 0 wird zu No, 1 wird zu Yes
data["EverTravelledAbroad"] = data["EverTravelledAbroad"].map({"No": 0, "Yes": 1})


#Visualisierung anhand der Variable Age --> Altersverteilung
figure = px.histogram(data, 
                      x = "Age", 
                      color = "TravelInsurance", 
                      title= "Altersabhängigkeit bei dem Kauf einer Reiseversicherung")
figure.show()

#Visualisierung anhand der Variable Employment Type
figure = px.histogram(data, 
                      x = "Employment Type", 
                      color = "TravelInsurance", 
                      title= "Beschäftigungsverhältnis bei dem Kauf einer Reiseversicherung")
figure.show()

#Visualisierung anhand der Variable AnnualIncome
figure = px.histogram(data, 
                      x = "AnnualIncome", 
                      color = "TravelInsurance", 
                      title= "Einkommen in Abhängigkeit zum Kauf einer Reiseversicherung")
figure.show()

#Visualisierung anhand der Variable Family
figure = px.histogram(data, 
                      x = "FamilyMembers", 
                      color = "TravelInsurance", 
                      title= "Reiseversicherung in Abhängigkeit der Anzahl an Familienmitgliedern")
figure.show()

#Visualisierung anhand der Variable Family
figure = px.histogram(data, 
                      x = "FrequentFlyer", 
                      color = "TravelInsurance", 
                      title= "Reiseversicherung in Abhängigkeit, ob man regelmäßig fliegt")
figure.show()

#Visualisierung anhand der Variable Family
figure = px.histogram(data, 
                      x = "EverTravelledAbroad", 
                      color = "TravelInsurance", 
                      title= "Reiseversicherung in Abhängigkeit, ob man schon mal im Ausland war")
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

#######################################################################
###########################DECISION TREE###############################
#######################################################################

#Deklaration und Initialisierung von DecisionTree Algorithmus
model = DecisionTreeClassifier()

#Laufzeitmessung Start DecisionTree Algorithmus
zeitanfangDecisionTree = time.time()

#Training des DecisionTree Algorithmus mit den Trainingsdaten
model.fit(xtrain, ytrain)

#Anhand der Testdaten können mithilfe des trainierten Algorithmus nun auch Vorhersagen getroffen werden
#predictions = model.predict(xtest)

#Wie viel kann der trainierte Algorithmus richtig vorhersagen?
print("=======================================================================================================")
print('Decision Tree Score: ' + str(model.score(xtest, ytest)))

#Laufzeitmessung Ende DecisionTree Algorithmus
zeitendeDecisionTree = time.time()

#Laufzeitmessung Berechnung und Ausgabe DecisionTree Algorithmus
print("=======================================================================================================")
print('Laufzeit für den Decision Tree: ' + str(zeitendeDecisionTree-zeitanfangDecisionTree))

#######################################################################
###########################RANDOM FOREST###############################
#######################################################################

#Deklaration und Initialisierung von RandomForest Algorithmus
model2 = RandomForestClassifier()

#Laufzeitmessung Anfang RandomForest Algorithmus
zeitanfangRandomForest = time.time()

#Training des RandomForest Algorithmus mit den Trainingsdaten
model2.fit(xtrain, ytrain.ravel())

#Anhand der Testdaten können mithilfe des trainierten Algorithmus nun auch Vorhersagen getroffen werden
#Zum Beispiel:
#predictions2 = model2.predict(xtest)

#Wie viel kann der trainierte Algorithmus richtig vorhersagen?
print("=======================================================================================================")
print('Random Forest Score: ' + str(model2.score(xtest, ytest)))

#Laufzeitmessung Ende RandomForest Algorithmus
zeitendeRandomForest = time.time()

#Laufzeitmessung Berechnung und Ausgabe RandomForest Algorithmus
print("=======================================================================================================")
print('Laufzeit für den Random Forest: ' + str(zeitendeRandomForest-zeitanfangRandomForest))

#Wichtigkeit der Variablen zur Vorhersage
feature_names = ["Age", "GraduateOrNot", 
                   "AnnualIncome", "FamilyMembers", 
                   "ChronicDiseases", "FrequentFlyer", 
                   "EverTravelledAbroad"]

importances = model2.feature_importances_
std = np.std([tree.feature_importances_ for tree in model2.estimators_], axis=0)

forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("")
fig.tight_layout()
plt.show()

print("=======================================================================================================")
print("==============================================USER INPUT===============================================")
print("=======================================================================================================")


#Usereingabe
userInputAge = int(input("Enter your Age: "))

userInputGraduateOrNot = input("Graduated? Yes or No: ")
if userInputGraduateOrNot=="Yes" or userInputGraduateOrNot=="yes":
    userInputGraduateOrNot = 1
else:
    userInputGraduateOrNot = 0

userInputAnnualIncome = float(input("AnnualIncome? Auf 5000 Euro genau gerundet: "))

userInputFamilyMembers = int(input("Amount of family members:  "))

userInputChronicDiseases = int(input("Chronic Diseases? Amount of chronic diseases: "))

userInputFrequentFlyer = input("Frequent Flyer? Yes or No: ")
if userInputFrequentFlyer=="Yes" or userInputFrequentFlyer=="yes":
    userInputFrequentFlyer = 1
else:
    userInputFrequentFlyer = 0

userInputEverTravelledAbroad = input("Ever Travelled Abroad? Yes or No: ")
if userInputEverTravelledAbroad=="Yes" or userInputEverTravelledAbroad=="yes":
    userInputEverTravelledAbroad = 1
else:
    userInputEverTravelledAbroad = 0

xpredict = np.array([userInputAge, userInputGraduateOrNot, userInputAnnualIncome, userInputFamilyMembers, userInputChronicDiseases, userInputFrequentFlyer, userInputEverTravelledAbroad]).reshape(1, -1)

print("Einschätzung des Decision Trees: " + model.predict(xpredict))
print("Einschätzung des Random Forest: " + model2.predict(xpredict))