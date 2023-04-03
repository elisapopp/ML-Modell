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

### Anwendungsfall 2: Vorhersagen von Reiseversicherungen

- Durch historische Daten soll vorhergesagt werden, ob eine Person eine Reiseversicherung abgeschlossen hat.
- Zudem sollen zwei verschiedene Algorithmen miteinander auf ihre Vorhersagbarkeit verglichen werden.
- Einerseits handelt es sich dabei um den typischen Entscheidungsbaum, der wie folgt importiert wird:
```java
  from sklearn.tree import DecisionTreeClassifier
```
- Andererseits wird der RandomForest als Vergleich dienen, der wie folgt importiert wird.
```java
  from sklearn.ensemble import RandomForestClassifier
```
- Für den Vergleich werden dieselben Daten verwendet, damit Unterschiede aufgrund verschiedener Input-Daten vermieden wird: https://raw.githubusercontent.com/amankharwal/Website-data/master/TravelInsurancePrediction.csv

---
---

## Schritt 1: Datenanalyse / Datenbereinigung

### Anwendungsfall 1: Spracherkennung

- TODO

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

#### Datenveränderung
- 

#### Datenüberprüfung
- Welche Datenattribute existieren in dem Datenset? Anzeigen der ersten 7 Einträge mithilfe: ```data.head(7)``` (https://www.w3schools.com/python/pandas/ref_df_head.asp)
- Welche Datentypen/allgemeine Informationen? Automatischer Print mithilfe: ```data.info()```
(https://www.w3schools.com/python/pandas/ref_df_info.asp)
- Existieren Null-Werte? ```print(data.isnull().sum())```
(https://www.w3schools.com/python/pandas/ref_df_isnull.asp)

---

## Schritt 2: Training der Algorithmen anhand der Daten

### Anwendungsfall 1: Spracherkennung

- TODO

### Anwendungsfall 2: Vorhersagen von Reiseversicherungen

- TODO

---

## Schritt 3: Vorhersagen


### Anwendungsfall 1: Spracherkennung

- TODO

### Anwendungsfall 2: Vorhersagen von Reiseversicherungen

- TODO


