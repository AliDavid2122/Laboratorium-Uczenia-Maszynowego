# -*- coding: utf-8 -*-
# =============================================================================
# Napisz skrypt, który implementuje algorytm k-means dla grupowania danych.
# Opis działań do realizacji:
# • Wygeneruj przykładowe dane (przy użyciu rozkładu normalnego) składające się
#   z dwóch zestawów punktów skupionych wokół dwóch różnych średnich.
# • Ustaw parametry algorytmu k-means, takie jak liczba klastrów i maksymalna 
#   liczba iteracji.
# • Zaimplementuj algorytm k-means, który przypisuje każdy punkt danych do 
#   najbliższego centroidu, aktualizuje położenia centroidów na podstawie 
#   przypisanych klastrów i powtarza ten proces przez określoną liczbę 
#   iteracji.
# • Po zakończeniu procesu grupowania, wizualizuj wyniki poprzez wygenerowanie 
#   wykresu punktowego danych, gdzie punkty są kolorowane na podstawie 
#   przypisanych klastrów, a centroidy są oznaczone na wykresie krzyżykami. 
#   W skrypcie należy również zawrzeć legendę, opisy osi i tytuł na wykresie.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# Funkcja do generowania przykladowych danych
def generate_data(num_points, num_clusters):
    data = []
    means = []
    for _ in range(num_clusters):
        mean = np.random.rand(2) * 10
        means.append(mean)
        cluster = np.random.randn(num_points // num_clusters, 2) + mean
        data.append(cluster)
    return np.concatenate(data), np.array(means)

# Funkcja do inicjalizacji centroidów
def initialize_centroids(data, num_clusters):
    indices = np.random.choice(len(data), num_clusters, replace=False)
    return data[indices]

# Funkcja do przypisywania punktów do klastrów
def assign_to_clusters(data, centroids):
    distances = np.linalg.norm(data[:, np.newaxis, :] - centroids, axis=2)
    return np.argmin(distances, axis=1)

# Funkcja do aktualizacji polozenia centroidów
def update_centroids(data, clusters, num_clusters):
    new_centroids = np.array([data[clusters == i].mean(axis=0) for i in range(num_clusters)])
    return new_centroids

# Funkcja do wizualizacji wyników grupowania
def plot_clusters(data, centroids, clusters):
    plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='viridis', alpha=0.7)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, color='red', label='Centroids')
    plt.title('K-Means Clustering')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.show()

# Parametry algorytmu k-means
num_points = 200
num_clusters = 2
num_iterations = 4

# Generowanie danych
data, original_means = generate_data(num_points, num_clusters)
print("Originalne centroidy:")
for i, mean in enumerate(original_means):
    print(f"Klaster {i+1}: {mean}")

# Inicjalizacja centroidów
centroids = initialize_centroids(data, num_clusters)

print("\n\nIteracja 0: (losowo wybrane centroidy)")
for i, mean in enumerate(centroids):
    print(f"Klaster {i+1}: {mean}")
    
# Algorytm k-means
for iteration in range(num_iterations):
    # Przypisanie punktów do klastrów
    clusters = assign_to_clusters(data, centroids)
    
    # Aktualizacja polozenia centroidów
    centroids = update_centroids(data, clusters, num_clusters)
    print(f"\n\nIteracja {iteration+1}:")
    for i, mean in enumerate(centroids):
        print(f"Klaster {i+1}: {mean}")

    # Wizualizacja wyników
    plot_clusters(data, centroids, clusters)

