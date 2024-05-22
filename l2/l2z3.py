# -*- coding: utf-8 -*-
# =============================================================================
# Napisz skrypt wykorzystujący metodę k najbliższych sąsiadów (k-NN) do 
# klasyfikacji danych.
# Opis działań do realizacji:
# • Wczytaj przykładowe dane (zbiór danych irysów) do klasyfikacji.
# • Znormalizuj dane, aby uniknąć wpływu cech o różnych skalach.
# • Podziel dane na zbiór treningowy i testowy.
# • Zbuduj model k-NN, ustalając liczbę sąsiadów k.
# • Dokonaj treningu modelu na danych treningowych.
# • Dokonaj klasyfikacji danych testowych przy użyciu wytrenowanego modelu 
#   k-NN.
# • Oceń skuteczność klasyfikacji, porównując przewidywane etykiety z 
#   rzeczywistymi etykietami testowymi.
# • Wyświetl wyniki oceny skuteczności klasyfikacji.
# • Przeprowadź wizualizację danych testowych i przewidywanych klas.
# =============================================================================


import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Wczytanie danych
iris = load_iris()
X = iris.data
y = iris.target

# Normalizacja danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Budowa modelu k-NN
k = 3  # liczba sąsiadów
knn = KNeighborsClassifier(n_neighbors=k)

# Trening modelu
knn.fit(X_train, y_train)

# Klasyfikacja danych testowych
y_pred = knn.predict(X_test)

# Ocenianie skuteczności klasyfikacji
accuracy = accuracy_score(y_test, y_pred)
print("Skuteczność klasyfikacji:", accuracy)

# Raport klasyfikacji
print("Raport klasyfikacji:")
print(classification_report(y_test, y_pred))

# Wizualizacja danych testowych i przewidywanych klas
plt.figure(figsize=(10, 6))

# Wizualizacja danych testowych
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', label='Test Data')

# Wizualizacja przewidywanych klas
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', marker='x', label='Predicted Classes')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Test Data and Predicted Classes')
plt.legend()
plt.show()
