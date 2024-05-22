# -*- coding: utf-8 -*-
# =============================================================================
# Napisz skrypt, który implementuje algorytm LDA (Linear Discriminant Analysis) 
# do klasyfikacji danych.
# Opis działań do realizacji:
# • Wczytaj dane ze zbioru Iris.
# • Zastosuj algorytm LDA do danych treningowych.
# • Dokonaj klasyfikacji danych testowych i ocen dokładność klasyfikacji.
# • Przedstaw wyniki wizualizacyjnie, np. poprzez wykresy rozrzutu danych z 
#   oznaczeniem przewidywanych klas.
# =============================================================================
# Źródła
# https://en.wikipedia.org/wiki/Linear_discriminant_analysis
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score

# Zastosowanie algorytmu LDA do danych treningowych

## Tworzy się instancję modelu LDA. W metodologii LDA, celem jest znalezienie 
## liniowych kombinacji cech, które najlepiej oddzielają różne klasy danych.
lda_model = LDA()

## Dopasowuje model LDA do danych treningowych X_train z ich etykietami 
## y_train. Model ten jest trenowany, aby nauczyć się liniowych 
## dyskryminacyjnych funkcji, które najlepiej separują klasy. Wynikowa 
## macierz X_train_lda zawiera przekształcone dane treningowe, gdzie wymiary 
## są redukowane do liczby komponentów, które najlepiej oddzielają klasy.
X_train_lda = lda_model.fit_transform(X_train, y_train)

# Klasyfikacja danych testowych

## Te dane testowe X_test są przekształcane przy użyciu nauczonych wcześniej 
## parametrów modelu LDA. Wynikowa macierz X_test_lda zawiera przekształcone
## dane testowe, zredukowane do wymiarów dyskryminacyjnych.
X_test_lda = lda_model.transform(X_test)

## Dane testowe są klasyfikowane przy użyciu nauczonego modelu LDA. Wyniki
## klasyfikacji są przypisywane do y_pred.
y_pred = lda_model.predict(X_test)

# Ocena dokładności klasyfikacji
accuracy = accuracy_score(y_test, y_pred)
print("Dokładność klasyfikacji:", accuracy)

# Wizualizacja wyników
plt.figure(figsize=(10, 6))

# Przewidywane klasy
plt.scatter(X_test_lda[:, 0], X_test_lda[:, 1], c=y_pred, cmap='viridis', marker='o', label='Przewidywane klasy')

# Prawdziwe klasy
plt.scatter(X_test_lda[:, 0], X_test_lda[:, 1], c=y_test, cmap='viridis', marker='x', s=100, label='Prawdziwe klasy')

plt.title('Wizualizacja wyników klasyfikacji LDA')
plt.xlabel('Składowa 1')
plt.ylabel('Składowa 2')
plt.legend()
plt.show()