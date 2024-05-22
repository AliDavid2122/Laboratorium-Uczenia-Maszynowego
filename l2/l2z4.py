# -*- coding: utf-8 -*-
# =============================================================================
# Napisz skrypt implementujący algorytm Random Forest do klasyfikacji danych 
# oraz wizualizujący pięć pierwszych drzew z lasu losowego.
# Opis działań do realizacji:
# • Wygeneruj dane do klasyfikacji (wykorzystaj odpowiedni zbiór danych lub 
#   wygeneruj losowe dane).
# • Podziel dane na zbiór treningowy i testowy
# • Zbuduj model Random Forest. Określ odpowiednią liczbę drzew i inne 
#   parametry.
# • Dokonaj klasyfikacji danych testowych przy użyciu wytrenowanego modelu 
#   Random Forest.
# • Użyj funkcji predict do klasyfikacji danych testowych.
# • Oceń skuteczność klasyfikacji.
# • Porównaj przewidywane etykiety z rzeczywistymi etykietami testowymi i 
#   oblicz dokładność klasyfikacji.
# • Wizualizuj pięć pierwszych drzew z Random Forest.
# • Wyświetl wyniki oceny skuteczności klasyfikacji.
# • Wyświetl dokładność klasyfikacji modelu.
# =============================================================================

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree

# Generowanie losowych danych do klasyfikacji
X, y = make_classification(n_samples=100, n_features=4, n_classes=2)

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Budowa modelu Random Forest
n_estimators = 100  # liczba drzew w lesie
max_depth = None  # maksymalna głębokość drzewa
random_forest = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
random_forest.fit(X_train, y_train)

# Klasyfikacja danych testowych
y_pred = random_forest.predict(X_test)

# Ocenianie skuteczności klasyfikacji
accuracy = accuracy_score(y_test, y_pred)
print("Dokładność klasyfikacji: {:.2f}%".format(accuracy * 100))

# Raport klasyfikacji
print("Raport klasyfikacji:")
print(classification_report(y_test, y_pred))

# Wizualizacja pięciu pierwszych drzew z Random Forest
for i in range(5):
    fig, ax = plt.subplots(dpi=900)
    tree.plot_tree(random_forest.estimators_[i], filled=True, feature_names=[f'Feature {i}' for i in range(20)])
    ax.set_title(f'Tree {i+1}')
    plt.show()
    