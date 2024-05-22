# -*- coding: utf-8 -*-
# =============================================================================
# Napisz skrypt, który generuje przykładowe dane oraz przeprowadza regresję 
# logistyczną.
# Opis działań do realizacji:
# • Wygeneruj 1000 losowych próbek cech X o rozmiarze 2 każda, korzystając z 
#   rozkładu jednostajnego. Dodatkowo, wygeneruj losowe etykiety klas 0 lub 1 
#   dla każdej próbki.
# • Dodaj kolumnę z jedynkami (bias) do macierzy cech X (pozwala to na 
#   uwzględnienie stałego składnika w modelu regresji logistycznej). 
#   Inicjalizuj parametry theta na początku jako wektor zerowy.
# • Zdefiniuj funkcję sigmoidalną, która będzie używana w regresji logistycznej 
#   do przekształcania wyników na prawdopodobieństwa.
# • Zdefiniuj funkcję kosztu (logistyczną entropię krzyżową), która będzie 
#   miarą różnicy między wartościami przewidywanymi przez model a rzeczywistymi
#   etykietami.
# • Zaimplementuj optymalizację parametrów theta za pomocą algorytmu 
#   gradientowego. Użyj stałego współczynnika uczenia i określ liczbę iteracji.
# • Przeprowadź klasyfikację na podstawie wyuczonego modelu. Oblicz dokładność
#   klasyfikacji.
# • Oblicz krzywą ROC i obszar pod nią (AUC), które są miarami wydajności 
#   klasyfikatora binarnego.
# • Wyświetl wyniki: dokładność klasyfikacji i AUC. Narysuj wykres krzywej ROC.
# Wynikowy skrypt powinien generować losowe dane, trenować model regresji 
# logistycznej, oceniać jego wydajność i prezentować wyniki oraz wykres krzywej
# ROC.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Generowanie losowych danych
# X = np.random.uniform(low=0, high=1, size=(1000, 2))
# y = np.random.randint(2, size=1000)

NUM_ELEMENTS = 1000

# Pierwszy zestaw punktów
data_class_0 = np.random.normal(loc=[1, 1], scale=[1, 1], size=(NUM_ELEMENTS // 2, 2))

# Drugi zestaw punktów
data_class_1 = np.random.normal(loc=[-1, -1], scale=[1, 1], size=(NUM_ELEMENTS // 2, 2))

# Przypisanie etykiet klas
labels_class_0 = np.zeros(NUM_ELEMENTS // 2)
labels_class_1 = np.ones(NUM_ELEMENTS // 2)

# Łączenie danych i etykiet
X = np.vstack([data_class_0, data_class_1])
y = np.hstack([labels_class_0, labels_class_1])


plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')
plt.title('Wykres punktowy danych X')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Labels')
plt.grid(True)
plt.show()

# Dodanie kolumny z jedynkami (bias)
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# Inicjalizacja parametrów theta jako wektor zerowy
theta = np.zeros(X_b.shape[1])

# Funkcja sigmoidalna
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Funkcja kosztu (logistyczna entropia krzyżowa)
def cost_function(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    return -1/m * np.sum(y*np.log(h) + (1-y)*np.log(1-h))

# Algorytm gradientowy do optymalizacji parametrów theta
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    
    for i in range(iterations):
        h = sigmoid(X.dot(theta))
        gradient = 1/m * X.T.dot(h - y)
        theta -= learning_rate * gradient
        cost_history[i] = cost_function(X, y, theta)
    
    return theta, cost_history

# Ustawienia algorytmu gradientowego
learning_rate = 0.1
iterations = 10000

# Trenowanie modelu
theta_optimized, cost_history = gradient_descent(X_b, y, theta, learning_rate, iterations)

# Klasyfikacja na podstawie wyuczonego modelu
y_pred = np.round(sigmoid(X_b.dot(theta_optimized)))

# Indeksacja logiczna dla dobrych i błędnych klasyfikacji
correct = (y == y_pred)
incorrect = ~correct

# Wykres punktowy dla poprawnie sklasyfikowanych elementów
plt.figure(figsize=(8, 6))
plt.scatter(X[correct, 0], X[correct, 1], c=y[correct], cmap='viridis', marker='o', alpha=0.1, edgecolors='k', label='Dobrze sklasyfikowane')
# Wykres punktowy dla błędnie sklasyfikowanych elementów
plt.scatter(X[incorrect, 0], X[incorrect, 1], c=y[incorrect], cmap='viridis', marker='s', edgecolors='k', label='Błędnie sklasyfikowane')

plt.title('Klasyfikacja na podstawie wyuczonego modelu')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
plt.colorbar(label='Labels')
plt.grid(True)
plt.show()

# Dokładność klasyfikacji
accuracy = np.mean(y_pred == y)

# Obliczanie krzywej ROC i obszaru pod nią (AUC)
fpr, tpr, _ = roc_curve(y, y_pred)
roc_auc = auc(fpr, tpr)

# Wyświetlanie wyników
print("Dokładność klasyfikacji:", accuracy)
print("Obszar pod krzywą ROC (AUC):", roc_auc)

# Rysowanie krzywej ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Krzywa ROC (obszar = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Krzywa ROC')
plt.legend(loc="lower right")
plt.show()
