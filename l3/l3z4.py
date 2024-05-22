# -*- coding: utf-8 -*-
# =============================================================================
# Napisz skrypt, który implementuje algorytm Softmax Regression do klasyfikacji 
# danych.
# Opis działań do realizacji:
# • Wczytaj dane ze zbioru Iris.
# • Zastosuj algorytm Softmax Regression do danych treningowych.
# • Dokonaj klasyfikacji danych testowych i ocen dokładność klasyfikacji.
# • Przedstaw wyniki wizualizacyjnie, np. poprzez wykresy rozrzutu danych z 
#   oznaczeniem przewidywanych klas.
# =============================================================================
# Źródła:
# https://d2l.ai/chapter_linear-classification/softmax-regression.html
# =============================================================================

# Wczytanie danych
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

# Normalizacja danych
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Podział danych na zbiór treningowy i testowy
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)

# Aby zaimplementować algorytm Softmax Regression do klasyfikacji danych za 
# pomocą biblioteki Scikit Learn, możemy skorzystać z klasy LogisticRegression 
# z ustawieniem parametru multi_class='multinomial', co odpowiada za stosowanie
# regresji softmax. 

# Import potrzebnych bibliotek
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Tworzenie i trenowanie modelu regresji logistycznej z użyciem softmax

## Tworzony jest obiekt modelu regresji logistycznej. Parametr 
## multi_class='multinomial' wskazuje, że chcemy przeprowadzić klasyfikację
## wieloklasową przy użyciu funkcji softmax. Parametr solver='lbfgs' oznacza,
## że używany będzie algorytm optymalizacyjny L-BFGS (Limited-memory 
## Broyden-Fletcher-Goldfarb-Shanno), który jest jednym z algorytmów 
## stosowanych w optymalizacji dla problemów regresji logistycznej.

############################### WYCIĄG Z HELP'a ###############################
### @param multi_class
### If the option chosen is 'ovr', then a binary problem is fit for each label.
### For 'multinomial' the loss minimised is the multinomial loss fit across the
### entire probability distribution, even when the data is binary. 
### 'multinomial' is unavailable when solver='liblinear'. 'auto' selects 'ovr'
### if the data is binary, or if solver='liblinear', and otherwise selects
### 'multinomial'.

################### sklearn.linear_model.LogisticRegression ###################
### For a multi_class problem, if multi_class is set to be “multinomial” the 
### softmax function is used to find the predicted probability of each class.
### Else use a one-vs-rest approach, i.e. calculate the probability of each
### class assuming it to be positive using the logistic function. and normalize
### these values across all the classes.
### Źródło: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

## Trenuje się model regresji logistycznej na danych treningowych X_train z 
## odpowiadającymi im etykietami y_train. Proces ten polega na dostosowaniu
## modelu do danych treningowych w taki sposób, aby model mógł przewidywać
## etykiety klas dla nowych danych.
model.fit(X_train, y_train)

# Dokonanie predykcji na danych testowych

## Dane testowe X_test są używane do przewidywania klas przez wytrenowany model
## regresji logistycznej. Wyniki przewidywania są przypisywane do zmiennej
## y_pred, która zawiera prognozy modelu dla danych testowych.
y_pred = model.predict(X_test)

# Obliczenie dokładności klasyfikacji
accuracy = accuracy_score(y_test, y_pred)
print("Dokładność klasyfikacji: {:.2f}%".format(accuracy * 100))

# Wizualizacja predykcji
plt.figure(figsize=(10, 6))

# Wizualizacja danych treningowych
for class_label in np.unique(y_train):
    indices = np.where(y_train == class_label)
    plt.scatter(X_train[indices, 0], X_train[indices, 1], label=f'Class {class_label}', alpha=0.8)

# Wizualizacja przewidywanych klas
for class_label in np.unique(y_pred):
    indices = np.where(y_pred == class_label)
    plt.scatter(X_test[indices, 0], X_test[indices, 1], marker='x', label=f'Predicted Class {class_label}', alpha=0.8)

plt.title('Wizualizacja predykcji klas')
plt.xlabel('Cecha 1')
plt.ylabel('Cecha 2')
plt.legend()
plt.grid(True)
plt.show()