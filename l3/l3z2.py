# -*- coding: utf-8 -*-
# =============================================================================
# Napisz skrypt, który implementuje algorytm SVM (Support Vector Machine) do 
# klasyfikacji danych.
# Opis działań do realizacji:
# • Wczytaj dane ze zbioru Iris.
# • Podziel dane na zbiór treningowy (70%) i testowy (30%) używając metody 
#   'Holdout'.
# • Zbuduj model SVM.
# • Dokonaj treningu modelu na danych treningowych.
# • Dokonaj klasyfikacji danych testowych i ocen dokładność klasyfikacji.
# • Przedstaw wyniki wizualizacyjnie, np. poprzez wykresy rozrzutu danych z 
#   oznaczeniem przewidywanych klas.
# =============================================================================
# Źródła:
# https://en.wikipedia.org/wiki/Support_vector_machine
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# https://www.iguazio.com/glossary/holdout-dataset/
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

# Importowanie niezbędnych bibliotek
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Budowa modelu SVM

## Tworzy się instancję modelu SVM z jądrem liniowym. SVM jest algorytmem 
## uczenia maszynowego stosowanym zarówno do klasyfikacji, jak i regresji. 
## Określając kernel='linear', używamy liniowego jądra SVM, które próbuje 
## znaleźć hiperpłaszczyznę, która najlepiej separuje klasy.
svm_model = SVC(kernel='linear')

# Trening modelu na danych treningowych

## Trenuje się model SVM na danych treningowych X_train z ich etykietami 
## y_train. Proces ten polega na znalezieniu optymalnej hiperpłaszczyzny 
## separującej klasy w przestrzeni cech.
svm_model.fit(X_train, y_train)

# Klasyfikacja danych testowych

## Dane testowe X_test są klasyfikowane przez nauczony model SVM. Klasyfikacje 
## są przewidywane dla danych testowych, a wyniki są przypisywane do zmiennej
## y_pred.
y_pred = svm_model.predict(X_test)

# Ocena dokładności klasyfikacji
accuracy = accuracy_score(y_test, y_pred)
print("Dokladnosc klasyfikacji:", accuracy)

# Wizualizacja wyników
plt.figure(figsize=(10, 6))

# Przewidywane klasy
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', marker='o', label='Przewidywane klasy')

# Prawdziwe klasy
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', marker='x', s=100, label='Prawdziwe klasy')

plt.title('Wizualizacja wyników klasyfikacji SVM')
plt.xlabel('Cecha 1 (znormalizowana)')
plt.ylabel('Cecha 2 (znormalizowana)')
plt.legend()
plt.show()