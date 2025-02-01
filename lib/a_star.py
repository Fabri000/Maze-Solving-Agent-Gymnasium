import heapq

def heuristic(a, b):
    """
    Calcola la distanza di Manhattan tra il punto a e il punto b.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar_limited_partial(maze, start, goal=None, max_depth=6):
    """
    Esegue una ricerca A* limitata in profondità per trovare un percorso
    da start a goal nel labirinto, senza superare max_depth passi.
    
    Se il percorso completo verso l'obiettivo non viene trovato entro il limite,
    la funzione restituisce comunque il percorso fino al nodo raggiunto con il
    valore massimo di g (ovvero il percorso più lungo effettuato).
    
    La rappresentazione del labirinto:
      0 -> muro (non percorribile)
      1 -> tessera percorribile
      2 -> obiettivo (percorribile)
    
    Parametri:
        maze (list of list di int): matrice 2D che rappresenta il labirinto.
        start (tuple): posizione iniziale come (riga, colonna).
        goal (tuple o None): posizione obiettivo come (riga, colonna). Se None,
                              l'algoritmo cerca la cella con valore 2.
        max_depth (int): numero massimo di passi (profondità) consentiti.
    
    Ritorna:
        list: percorso (completo o parziale) da start a goal (o al nodo migliore raggiunto)
              come lista di tuple (riga, colonna).
    """
    rows, cols = len(maze), len(maze[0])
    
    # Se non viene fornito l'obiettivo, cerchiamo la cella con valore 2
    if goal is None:
        for r in range(rows):
            for c in range(cols):
                if maze[r][c] == 2:
                    goal = (r, c)
                    break
            if goal is not None:
                break
        if goal is None:
            print("Non è stato trovato alcun obiettivo (cella con valore 2) nel labirinto.")
            return None

    # Inizializziamo la coda a priorità (open_set) e le strutture di supporto
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), start))
    
    came_from = {}          # Per ricostruire il percorso
    g_score = {start: 0}    # Costo dal nodo start al nodo corrente
    f_score = {start: heuristic(start, goal)}  # f = g + h
    
    # Variabile per tenere traccia del nodo raggiunto con il massimo numero di passi (g)
    best_candidate = start
    best_g = g_score[start]
    
    while open_set:
        current_f, current = heapq.heappop(open_set)
        
        # Aggiorna best_candidate se abbiamo raggiunto un nodo con g maggiore
        if g_score[current] > best_g:
            best_g = g_score[current]
            best_candidate = current

        # Se raggiungiamo l'obiettivo, ricostruiamo e restituiamo il percorso completo.
        if current == goal:
            return reconstruct_path(came_from, current)
        
        # Se il costo per arrivare qui ha raggiunto il limite massimo, non espandiamo ulteriormente.
        if g_score[current] >= max_depth:
            continue
        
        # Esploriamo i vicini (su, giù, sinistra, destra)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            
            # Verifica che il vicino sia entro i limiti del labirinto
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                # Verifica che il vicino sia percorribile (1 o 2)
                if maze[neighbor[0]][neighbor[1]] != 0:
                    tentative_g = g_score[current] + 1  # ogni passo costa 1
                    
                    # Se il costo supera il limite, non consideriamo questo vicino.
                    if tentative_g > max_depth:
                        continue
                    
                    # Se il percorso verso il vicino è migliore, lo aggiorniamo.
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f = tentative_g + heuristic(neighbor, goal)
                        f_score[neighbor] = f
                        heapq.heappush(open_set, (f, neighbor))
    
    # Se l'obiettivo non è stato raggiunto, restituiamo il percorso fino al best_candidate.
    return reconstruct_path(came_from, best_candidate)

def reconstruct_path(came_from, current):
    """
    Ricostruisce il percorso a partire dal nodo current usando il dizionario came_from.
    
    Parametri:
        came_from (dict): Dizionario che mappa ogni nodo al suo predecessore.
        current (tuple): Nodo di partenza per la ricostruzione (goal o best_candidate).
    
    Ritorna:
        list: Percorso come lista di nodi (tuple) dal nodo iniziale a quello corrente.
    """
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

