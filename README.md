# Maschinelles Lernen: Portfolio

---

## Anwendungsfälle im Allgemeinen erklärt

### Anwendungsfall 1: Spracherkennung

- Der Algorithmus soll eine User-Eingabe sprachlich korrekt erkennen und ausgeben.
- Verwendung einer Vektorisierung: CountVectorizer (siehe weiter unten)
```java
  from sklearn.feature_extraction.text import CountVectorizer
```
- Verwendung des MultinomialNB-Algorithmus
- Der Algorithmus wird im Code wie folgt aufgerufen:
```java
  from sklearn.naive_bayes import MultinomialNB
```
- Die Daten sind unter folgender URL verfügbar: https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv
- Orienierung an: https://practicaldatascience.co.uk/machine-learning/how-to-create-a-naive-bayes-text-classification-model-using-scikit-learn

### Anwendungsfall 2: Vorhersagen von Reiseversicherungen

- Durch historische Daten soll vorhergesagt werden, ob eine Person eine Reiseversicherung abgeschlossen hat.
- Zudem sollen zwei verschiedene Algorithmen miteinander auf ihre Vorhersagbarkeit verglichen werden.
- Einerseits handelt es sich dabei um den typischen Entscheidungsbaum, der wie folgt importiert wird:
```java
  from sklearn.tree import DecisionTreeClassifier
```
- https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
- Andererseits wird der RandomForest als Vergleich dienen, der wie folgt importiert wird.
```java
  from sklearn.ensemble import RandomForestClassifier
```
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- Für den Vergleich werden dieselben Daten verwendet, damit Unterschiede aufgrund verschiedener Input-Daten vermieden wird: https://raw.githubusercontent.com/amankharwal/Website-data/master/TravelInsurancePrediction.csv

---
---

## Schritt 1: Datenanalyse / Datenbereinigung

### Anwendungsfall 1: Spracherkennung

#### Datenüberprüfung

- Bei der Analyse der Daten werden diese zunächst mithilfe der Methoden ```data.isnull().sum()``` nach fehlenden Werten durchsucht und die Anzahl ausgegeben.

- Durch die Verwendung der Methode ```data["language"].value_counts()``` wird die Anzahl  der vorkommenden Werte in der Spalte ermittelt. Das Ergebnis zeigt, dass es sich bei diesem Datenset um ein Klassifikationsproblem mit mehreren Klassen handelt.

#### Datenveränderung

- Hier noch erläutern, wenn die daten modifiziert werden.

### Anwendungsfall 2: Vorhersagen von Reiseversicherungen

#### Visualisierung 

- Um die Daten aus dem Datenset gut visualisieren zu können, wurde das Modul plotly Express (https://plotly.com/python/plotly-express/) verwendet und im Code wie folgt importiert:
```java
  import plotly.express as px
```
- Mithilfe der Methode ```px.histogram(parameters)``` kann ein visuelles Histogramm erzeugt werden.
- Als Parameter dienen:
  - ```data```: die Daten (aus dem zuvor genannten Datenset), 
  - ```x```: x-Variable des Histogramms, 
  - ```color```: Unterscheidung mithilfe von Farben anhand der mitgegebenen Zielvariablen
  - ```title```: Titel des Histogramms

#### Datenüberprüfung
- Welche Datenattribute existieren in dem Datenset? Anzeigen der ersten 7 Einträge mithilfe: ```data.head(7)``` (https://www.w3schools.com/python/pandas/ref_df_head.asp)
- Welche Datentypen/allgemeine Informationen? Automatischer Print mithilfe: ```data.info()```
(https://www.w3schools.com/python/pandas/ref_df_info.asp)
- Existieren Null-Werte? ```print(data.isnull().sum())```
(https://www.w3schools.com/python/pandas/ref_df_isnull.asp)

#### Datenveränderung
- Unrelevante Daten werden gelöscht. In dem Fall spiel der Parameter Unnamed keine Rolle und wird mithilfe von ```data.drop(columns=["Unnamed: 0"], inplace=True)``` aus dem Datenset gelöscht. 
- Verwendete Parameter:
  - Columns legt fest, welche Daten verändert werden sollen,
  - Inplace sagt dabei aus, dass keine Kopie erstellt wird und stattdessen das Datenset verändert wird
- Zur besseren Lesbarkeit der Daten wird die Variable TravelInsurance nicht mehr mit 0 oder 1 angegeben, sondern gemappt, indem einer 0 der Text "Besitzt keine Reiseversicherung" und einer 1 der Text "Besitzt eine Reiseversicherung" zugeordnet wird. Dies erleichtert auch das Verständnis der visualisierten Daten.
- Auch bei den Variablen GraduateOrNot, FrequentFlyer und EverTravelledAbroad wird ein Mapping durchgeführt mit jeweils 0 --> No, 1 --> Yes

---

## Schritt 2: Training der Algorithmen anhand der Daten

### Anwendungsfall 1: Spracherkennung

#### Vorbereitung der Daten

- da Modelle für maschinelles Lernen keinen Text verwenden, werden die Texte zunächst in Vektoren umgewandelt. Dazu wird die Vorbereitungstechnik Count Vectorization verwendet. Dies wird mithilfe des CountVectorizer-Moduls realisiert:

```java
  from sklearn.feature_extraction.text import CountVectorizer
```
- CountVectorizer zählt wie häufig ein Wort in einem Dokument vorkommt. Jede Koordinate repräsentiert ein Wort. Durch die Methode ```cv.fit_transform(x)``` wird das Ergebnis als Bag-of-Words-Modell dargestellt. Dabei wird jedes Dokument als Sammlung der darin enthaltenen Wörter und deren Häufigkeit dargstellt. Wenn die Koordinate ```(0,5) 1``` lautet, dann heißt dies, dass das Dokument an der Stelle Null ein bestimmtes Wort ein mal enthält (maybe, nicht ganz sicher).

- Durch den Befehl und die Funktion ```X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)``` werden die mitgegebenen Arrays in zufällige Testteilmengen aufgeteilt. Bei dem Array y handelt es sich um das Array, welches die Sprache beinhaltet, bei dem Array X und das Array, welches durch den CountVectorizer erstellt wurde. Die test_size beschreibt den Anteil der Datensetz, welcher in die Testaufteilung aufgenommen werden soll. Dieser wurde auf 0,2 festgelegt, da dies eine ideale Aufteilung für Traing und Test sei. Der Parameter random_state führt dazu, dass das ergebnis reproduzierbar ist, indem eine festgelegte Zahl in die Funktion gegeben wird. Durch den Paramter stratify kann sicher gestellt werden, dass die Verteilung der Daten auf die Teilmengen gleichmäßig ist.

#### Algorithmus trainieren und testen

- Der MultinomialNB-Klassifikator eignet sich für die Klassifizierung mit ganzzahligen Merkmalszahlen, wie die aus dem CountVectorizer-Prozess erstellten Daten. Der Methode ```model.fit(X_train,y_train)``` werden die zuvor erstellen Testteilmengen X_train und y_train mitgegeben, um das Model so zu trainieren, dass es die Sprache aus den Vektoren der Texte vorhersagen kann.

- Zum Testen des Models wird eine score Methode verwendet. 


- Quelle: https://scikit-learn.org/stable/modules/model_evaluation.html

### Anwendungsfall 2: Vorhersagen von Reiseversicherungen

- TODO

---

## Schritt 3: Vorhersagen


### Anwendungsfall 1: Spracherkennung

- TODO

### Anwendungsfall 2: Vorhersagen von Reiseversicherungen

- TODO


