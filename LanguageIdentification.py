import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# dataset wird über einen Link eingelesen
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv")

# Analyse des Datensets: 
# Anzahl an fehlenden Werten erkennen & ausgeben
# die Funktion value_counts() zählt die Anzahl der vorkommenden Werte, den diese Spalte annimmt
print(data.isnull().sum())
print(data["language"].value_counts())

# Hier könnte noch analysiert werden, ob doppelte Wörter in den verschiedenen Sprachen vorkommen. Diese werden eleminiert.

# Für Beide Spalten des datasets wird ein Array erstellt
x = np.array(data["Text"])
y = np.array(data["language"])

# Test- und Trainingsdaten erzeugen:
# Konvertierung eines Datensetz in eine Matrix von Token Zählungen, z.B. an Stelle (0,3) kommt ein Wort x mal vor
cv = CountVectorizer()
# Häufigkeit, wie oft ein Wort im Text vorkommt
X = cv.fit_transform(x)
# Arrays werden in Zufällige Testteilmengen aufgeteilt
# y,X : Folgen von Indexen gleicher Länge; test_size: Anteil der Datensets, welcher in die Testaufteilung aufgenommen werden soll; 
# random_state: Steuerung der Mischung vor Teilung, int für die reproduzierbare Ausgabe
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42, stratify=y)

# MultinomialNB-Klassifikator eignet sich für Klassifizierung mit diskreten Merkmalen (ganzzahlige Merkmalszahlen)
# Quelle: https://practicaldatascience.co.uk/machine-learning/how-to-create-a-naive-bayes-text-classification-model-using-scikit-learn
model = MultinomialNB()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))

user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = model.predict(data)
print(output)
