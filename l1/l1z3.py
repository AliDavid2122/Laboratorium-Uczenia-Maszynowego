# -*- coding: utf-8 -*-
# =============================================================================
# Napisz skrypt, który przeprowadzi analizę danych zestawu Iris. Skrypt 
# powinien zawierać następujące kroki:
# Opis działań do realizacji:
# • Wczytaj zestaw danych Iris. Przypisz cechy do zmiennej `X` oraz etykiety 
#   klas do zmiennej `y`.
# • Wyświetl podstawowe informacje o danych, takie jak liczba przykładów, 
#   liczba cech i unikalne klasy.
# • Stwórz wykres, który wizualizuje długość i szerokość działki kielicha 
#   kwiatu dla różnych klas.
# • Usuń brakujące wartości, tj. wiersze zawierające braki danych. Znormalizuj 
#   cechy do zakresu [0,1].
# • Podziel dane na zbiór treningowy (70%) i testowy (30%) przy użyciu funkcji 
#   `cvpartition`.
# • Wyświetl podsumowanie po przygotowaniu danych, w tym liczbę wierszy po 
#   usunięciu brakujących wartości oraz liczbę wierszy w zbiorach treningowym i
#   testowym.
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Kroki 1-2: Wczytanie danych i wyświetlenie informacji o danych
url = "iris.data"
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
iris_data = pd.read_csv(url, header=None, names=column_names, na_values='?')
X = iris_data.drop("class", axis=1)
y = iris_data["class"]

print("Podstawowe informacje o danych:")
print(iris_data.info())

# Kroki 3: Wykres długości i szerokości działki kielicha dla różnych klas
plt.figure(figsize=(10, 6))
for flower_class in iris_data["class"].unique():
    subset = iris_data[iris_data["class"] == flower_class]
    plt.scatter(subset["sepal_length"], subset["sepal_width"], label=flower_class)

plt.title("Długość vs Szerokość działki kielicha")
plt.xlabel("Długość działki kielicha")
plt.ylabel("Szerokość działki kielicha")
plt.legend()
plt.show()

# Kroki 4: Usunięcie brakujących wartości i normalizacja cech
iris_data.dropna(inplace=True)
scaler = MinMaxScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Kroki 5: Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, stratify=y)

# Kroki 6: Wyświetlenie podsumowania danych po przygotowaniu
print("\nPodsumowanie po przygotowaniu danych:")
print("Liczba wierszy po usunięciu brakujących wartości:", len(iris_data))
print("Liczba wierszy w zbiorze treningowym:", len(X_train))
print("Liczba wierszy w zbiorze testowym:", len(X_test))
