# -*- coding: utf-8 -*-
# =============================================================================
# Napisz skrypt, który generuje przykładowe dane, dzieli je na zbiór treningowy
# i testowy, buduje prosty model sieci neuronowej, klasyfikuje dane testowe za
# pomocą sieci neuronowej i ocenia skuteczność klasyfikacji.
# Opis działań do realizacji:
# • Wygeneruj przykładowe dane składające się z dwóch zestawów punktów 
#   skupionych wokół dwóch różnych średnich, korzystając z rozkładu normalnego.
# • Przypisz etykiety klas dla wygenerowanych danych (pierwszy zestaw punktów 
#   oznacz jako klasa "1", a drugi zestaw jako klasa "0").
# • Podziel dane na zbiór treningowy (80%) i testowy (20%).
# • Zbuduj prosty model sieci neuronowej z jedną warstwą ukrytą zawierającą 10 
#   neuronów.
# • Dokonaj treningu sieci neuronowej na danych treningowych, włączając okno 
#   pokazujące postęp uczenia.
# • Dokonaj klasyfikacji danych testowych przy użyciu wytrenowanej sieci 
#   neuronowej, w opcji z zaokrągleniem wyników klasyfikacji do 0 lub 1 i w 
#   opcji bez zaokrąglenia.
# • Oceń skuteczność klasyfikacji poprzez porównanie przewidywanych etykiet z 
#   rzeczywistymi etykietami testowymi.
# • Wyświetl informacje o dokładności klasyfikacji.
# =============================================================================

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

NUM_ELEMENTS = 1000

# Pierwszy zestaw punktów skupiony wokół średniej (2, 2)
data_class_1 = np.random.normal(loc=[2, 2], scale=[1, 1], size=(NUM_ELEMENTS // 2, 2))

# Drugi zestaw punktów skupiony wokół średniej (-2, -2)
data_class_0 = np.random.normal(loc=[-2, -2], scale=[1, 1], size=(NUM_ELEMENTS // 2, 2))

# Przypisanie etykiet klas
labels_class_1 = np.ones(NUM_ELEMENTS // 2)
labels_class_0 = np.zeros(NUM_ELEMENTS // 2)

# Łączenie danych i etykiet
data = np.vstack([data_class_1, data_class_0])
labels = np.hstack([labels_class_1, labels_class_0])

# Podział danych na zbiór treningowy (80%) i testowy (20%)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels)

# Budowa modelu sieci neuronowej z jedną warstwą ukrytą zawierającą 10 neuronów
model = MLPClassifier(hidden_layer_sizes=(10,10,10,10,10,10,10,10,10,10), activation='relu', solver='adam', max_iter=1000, verbose=True)

# Trening modelu
model.fit(X_train, y_train)

# Klasyfikacja danych testowych
predictions_rounded = model.predict(X_test)
predictions_continuous = model.predict_proba(X_test)[:, 1]

# Ocenia skuteczność klasyfikacji
accuracy = accuracy_score(y_test, predictions_rounded)

# Wyświetlenie informacji o dokładności klasyfikacji
print(f"Dokładność klasyfikacji: {accuracy}")

from matplotlib.colors import LinearSegmentedColormap
colors = [ (1.0, 1.0, 0.0), (0.0, 0.0, 0.5)]  # yellow to navy 
yellow_to_navy_cmap = LinearSegmentedColormap.from_list('yellow_to_navy', colors)


# Dane uczące
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='yellow',  alpha=0.75, marker='x', label='Dane uczące klasa 0')
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='navy', alpha=0.75, marker='x', label='Dane uczące klasa 1')
plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c='yellow',  alpha=0.75, marker='o', label='Dane testowe klasa 0')
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c='navy',  alpha=0.75, marker='o', label='Dane testowe klasa 1')
plt.title('Zestaw danych')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(bbox_to_anchor=(0.5, -0.4), loc='lower center', ncol=2)
plt.show()

plt.scatter(X_test[:, 0], X_test[:, 1], c=predictions_continuous, cmap=yellow_to_navy_cmap, alpha=0.75)
plt.title('Dane testowe bez zaokrąglenia')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.colorbar(label='Prawdopodobieństwo przynależności do klasy "1"')
plt.show()


misclassified_class_0 = X_test[(predictions_rounded != y_test) & (y_test == 0)]
correctly_classified_class_0 = X_test[(predictions_rounded == y_test) & (y_test == 0)]
plt.scatter(misclassified_class_0[:, 0], misclassified_class_0[:, 1], c='yellow',  marker='x', alpha=1, label='Błędnie sklasyfikowane klasy 0')
plt.scatter(correctly_classified_class_0[:, 0], correctly_classified_class_0[:, 1], c='yellow', marker='o', alpha=0.75, label='Prawidłowo sklasyfikowane klasy 0')
misclassified_class_1 = X_test[(predictions_rounded != y_test) & (y_test == 1)]
correctly_classified_class_1 = X_test[(predictions_rounded == y_test) & (y_test == 1)]
plt.scatter(misclassified_class_1[:, 0], misclassified_class_1[:, 1], c='navy', marker='x', alpha=1, label='Błędnie sklasyfikowane klasy 1')
plt.scatter(correctly_classified_class_1[:, 0], correctly_classified_class_1[:, 1], c='navy', marker='o', alpha=0.75, label='Prawidłowo sklasyfikowane klasy 1')

plt.title('Dane testowe z zaokrągleniem')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(bbox_to_anchor=(0.5, -0.4), loc='lower center', ncol=2)
plt.show()