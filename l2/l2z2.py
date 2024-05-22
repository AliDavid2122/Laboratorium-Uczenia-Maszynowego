# -*- coding: utf-8 -*-
# =============================================================================
# Napisz skrypt w wykorzystujący drzewo decyzyjne do przewidywania 
# choroby serca na podstawie pięciu cech pacjentów: tetno, poziom_cukru, 
# c_krwi_sk, c_krwi_ro, wiek.
# Opis działań do realizacji:
# • Wygeneruj losowe dane dla 200 pacjentów (tętno - losowe wartości z zakresu
#   60-100, poziom_cukru - losowe wartości z zakresu 70-130, c_krwi_sk - losowe
#   wartości z zakresu 90-140, c_krwi_ro - losowe wartości z zakresu 70-120, 
#   wiek - losowe wartości z zakresu 25-80).
# • Ustal etykietę choroby serca (0 - brak choroby, 1 - obecność choroby) na 
#   podstawie warunków (przykładowo : tetno > 80, poziom_cukru > 100, 
#   c_krwi_sk > 120, c_krwi_sk>110, wiek > 50.
# • Podziel dane na zbiór treningowy (80%) i testowy (20%).
# • Trenuj drzewo decyzyjne na zbiorze treningowym.
# • Zwizualizuj drzewo decyzyjne.
# • Dokonaj klasyfikacji na zbiorze testowym i ocen dokładność klasyfikacji.
# • Przeprowadź predykcję dla pojedynczego pacjenta o podanych cechach i 
#   wyświetl wynik.
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Generowanie losowych danych dla 200 pacjentów
n_patients = 200
tetno = np.random.randint(60, 101, n_patients)
poziom_cukru = np.random.randint(70, 131, n_patients)
c_krwi_sk = np.random.randint(90, 141, n_patients)
c_krwi_ro = np.random.randint(70, 121, n_patients)
wiek = np.random.randint(25, 81, n_patients)

# Ustalenie etykiety choroby serca (0 - brak choroby, 1 - obecność choroby)
labels = ((tetno > 80) & (poziom_cukru > 100) & ((c_krwi_sk > 120) | (c_krwi_ro > 110)) & (wiek > 50)).astype(int)

# Podział danych na zbiór treningowy i testowy
X = np.column_stack((tetno, poziom_cukru, c_krwi_sk, c_krwi_ro, wiek))
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)

# Trenowanie drzewa decyzyjnego na zbiorze treningowym
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

# Wizualizacja drzewa decyzyjnego
plt.figure(dpi=300)
plot_tree(decision_tree, filled=True, feature_names=["Tętno", "Poziom cukru", "C. krwi sk", "C. krwi ro", "Wiek"], class_names=["Brak choroby", "Obecność choroby"])
plt.show()

# Klasyfikacja na zbiorze testowym i ocena dokładności klasyfikacji
y_pred = decision_tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Dokładność klasyfikacji: {:.2f}%".format(accuracy * 100))

# Predykcja dla pojedynczego pacjenta o podanych cechach
patient_features = np.array([[75, 110, 130, 100, 60]])  # Przykładowe cechy pacjenta
prediction = decision_tree.predict(patient_features)
print("Predykcja dla pojedynczego pacjenta:", "Obecność choroby" if prediction[0] == 1 else "Brak choroby")
