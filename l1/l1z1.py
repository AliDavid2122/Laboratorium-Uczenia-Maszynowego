# -*- coding: utf-8 -*-
# =============================================================================
# Napisz skrypt, który generuje przykładowe dane oraz przeprowadza regresję 
# liniową.
# Opis działań do realizacji:
# • Wygeneruj 100 losowych wartości cechy X z zakresu od 0 do 10 oraz wartości
#   docelowe y z liniową zależnością od X dodając szum gaussowski, co oznacza,
#   że y ma rozkład normalny z dodanym szumem.
# • Dodaj kolumnę z jedynkami (bias) do macierzy cech X (pozwala to na 
#   uwzględnienie stałego składnika w modelu liniowym). Oblicz parametry θ 
#   (theta) przy użyciu metody najmniejszych kwadratów. parametry te 
#   reprezentują współczynniki prostej liniowej, która jest dopasowywana do 
#   danych.
# • Wyświetl obliczone parametry θ.
# • Oblicz wartości przewidywane y_pred na podstawie danych X i parametrów θ.
# • Oblicz i wyświetl błąd średniokwadratowy (RMSE) i błąd średni absolutny 
#   (MAE) między wartościami rzeczywistymi y a przewidywanymi y_pred. Dokonaj 
#   interpretacji otrzymanych wyników.
# • Wygeneruj wykres punktowy danych i krzywej reprezentującej dopasowany model
#   regresji liniowej. Na wykresie zaznacz legendę oraz opisy osi i tytuł.
# Wynikowy skrypt powinien generować dane, obliczać parametry regresji 
# liniowej, prezentować wyniki i wykres wizualizujący dopasowany model do 
# danych.
# =============================================================================


import numpy as np
import matplotlib.pyplot as plt

# Ustawienie ziarna dla reprodukowalności wyników
# np.random.seed(42)

# Generowanie danych
X = 10 * np.random.rand(100, 1)
y = 2 * X + 1 + np.random.randn(100, 1)
# Dodanie kolumny z jedynkami (bias)
X_b = np.c_[np.ones((100, 1)), X]

# Obliczenie parametrów theta przy użyciu metody najmniejszych kwadratów:
# X_b.T @ X_b: 
#   Tutaj obliczamy iloczyn macierzowy transpozycji macierzy X_b (macierz 
#   rozszerzonych cech X, w której pierwsza kolumna składa się z jednostek) i 
#   macierzy X_b. Ten krok daje nam macierz kwadratową, która jest częścią 
#   obliczeń do estymacji parametrów.
# np.linalg.inv(...): 
#   Następnie stosujemy funkcję inv z biblioteki numpy.linalg, aby obliczyć 
#   odwrotność tej macierzy kwadratowej. To jest etap kluczowy, ponieważ 
#   odwrotność macierzy jest używana w metodzie najmniejszych kwadratów, aby 
#   uzyskać optymalne wartości parametrów.
# ... @ X_b.T: 
#   Teraz obliczamy iloczyn tej odwrotności macierzy przez transpozycję 
#   macierzy X_b.
# ... @ y: 
#   Na koniec obliczamy iloczyn tego wyniku przez wektor wartości docelowych y.
#   To daje nam ostateczne wartości parametrów modelu liniowego (wartości θ), 
#   które minimalizują sumę kwadratów różnic między wartościami przewidywanymi 
#   a rzeczywistymi.
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

# https://www.naukowiec.org/wiedza/statystyka/metoda-najmniejszych-kwadratow_733.html
X_mean = np.mean(X)
y_mean = np.mean(y)
N = np.size(X)
b = (np.sum(X * y) - N * X_mean * y_mean) / (np.sum(X * X) - N * X_mean * X_mean)
a = y_mean - b * X_mean
theta_best2 = np.array(([a], [b]))

# https://www.statystyczny.pl/regresja-liniowa/
X_dif = X - X_mean
y_dif = y - y_mean
A = np.sum(X_dif * y_dif) / np.sum(X_dif * X_dif)
B = y_mean - A * X_mean
theta_best3 = np.array(([B], [A]))


# Wyświetlenie parametrów theta
print("Wartości parametrów theta:")
print(theta_best)

# Obliczenie wartości przewidywanych
y_pred = X_b.dot(theta_best)

# Obliczenie błędów
rmse = np.sqrt(np.mean((y_pred - y)**2))
mae = np.mean(np.abs(y_pred - y))

# Wyświetlenie błędów
print(f"\nBłąd średniokwadratowy (RMSE): {rmse}")
print(f"Błąd średni absolutny (MAE): {mae}")

# Wygenerowanie wykresu
plt.scatter(X, y, label='Dane rzeczywiste')
plt.plot(X, y_pred, 'r-', label='Model regresji liniowej')
plt.xlabel('Cecha X')
plt.ylabel('Wartość docelowa y')
plt.title('Regresja Liniowa - Dopasowany Model')
plt.legend()
plt.show()
