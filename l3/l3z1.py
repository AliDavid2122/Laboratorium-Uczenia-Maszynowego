# -*- coding: utf-8 -*-
# =============================================================================
# Napisz skrypt, który wykorzystuje algorytm PCA (Principal Component Analysis)
# do redukcji wymiarowości danych. 
# Opis działań do realizacji:
# • Wczytaj zestaw danych Iris
# • Znormalizuj dane.
# • Zastosuj algorytm PCA, wybierając dwie składowe główne.
# • Przedstaw wyniki wizualizacyjnie, porównując oryginalne dane z danymi po 
# redukcji wymiarowości.
# =============================================================================
# Źródła:
# https://en.wikipedia.org/wiki/Principal_component_analysis
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

# Zastosowanie algorytmu PCA
from sklearn.decomposition import PCA

## Tworzy się instancję obiektu PCA. Parametr n_components=2 oznacza, że 
## chcemy zredukować wymiarowość danych do dwóch wymiarów, co oznacza, że 
## uzyskamy dwie główne składowe (komponenty).
pca = PCA(n_components=2)

## Dane X_scaled są poddane transformacji za pomocą PCA. Metoda fit_transform()
## najpierw dopasowuje model PCA do danych X_scaled, a następnie przekształca
## te dane do nowej przestrzeni cech opisanej przez wybrane komponenty główne.
## Wynikowe dane są przechowywane w X_pca.
X_pca = pca.fit_transform(X_scaled)

# Importowanie bibliotek do wizualizacji
import matplotlib.pyplot as plt
import seaborn as sns

# Wizualizacja oryginalnych danych
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=y, palette='viridis', s=100)
plt.title('Oryginalne dane (2 składowe)')
plt.xlabel('Składowa 1')
plt.ylabel('Składowa 2')
plt.legend(loc='best')
plt.show()

# Wizualizacja danych po redukcji wymiarowości
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis', s=100)
plt.title('Dane po redukcji wymiarowości (2 składowe)')
plt.xlabel('Składowa 1')
plt.ylabel('Składowa 2')
plt.legend(loc='best')
plt.show()