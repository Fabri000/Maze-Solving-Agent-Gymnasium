import numpy as np

def laplacian_grid(n):

    size = int(np.sqrt(n))  # Lato della griglia (3 per n=9)
    L = np.zeros((n, n))  # Matrice Laplaciana n x n

    for x in range(size):
        for y in range(size):
            i = x * size + y  # Indice del nodo nella matrice

            # Collegamento orizzontale (a destra)
            if y < size - 1:
                j = i + 1
                L[i, j] = -1
                L[j, i] = -1
                L[i, i] += 1  # Aggiungiamo al grado del nodo
                L[j, j] += 1  # Aggiungiamo al grado del vicino

            # Collegamento verticale (in basso)
            if x < size - 1:
                j = i + size
                L[i, j] = -1
                L[j, i] = -1
                L[i, i] += 1  # Aggiungiamo al grado del nodo
                L[j, j] += 1  # Aggiungiamo al grado del vicino
    return L


for i in range(3,13,2):
    L = laplacian_grid(i*i)
    tmp = np.delete(L, 0, 0)
    tmp = np.delete(tmp, 0, 1)

    print(f"Size {i}x{i}| Number of mazes {int(np.linalg.det(tmp))}")