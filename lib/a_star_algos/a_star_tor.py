import heapq

def heuristic(a, b,rows,cols):
    """
    Manhattan distance between two points in a toroidal space.        
    """
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])

    dx = min(dx, rows - dx)
    dy = min(dy, cols - dy)
    return dx + dy


def astar_limited_partial(maze, start, goal, max_depth=1e6):
    """
    Performs a depth-limited A* search to find a path from the start to the goal in the maze, without exceeding max_depth steps.

    If the complete path to the goal is not found within the limit, the function still returns the path up to the node reached with the highest g value (i.e., the longest path explored).

    Maze representation:
    0 → wall (not traversable)
    1 → walkable tile
    2 → goal (traversable)
        
    Args:
        maze (array):matrix representation of the array.
        start (tuple): starting position.
        goal (tuple): goal position. If None, find the position in the matrix associated to the value 2.
        max_depth (int): max depth for the a* algorithm.
    
    Returns:
        list: path from start to goal if close enough, or a partial path.
    """
    rows, cols = len(maze), len(maze[0])

    # Inizializziamo la coda a priorità (open_set) e le strutture di supporto
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal,rows,cols), start))
    
    came_from = {}          # Per ricostruire il percorso
    g_score = {start: 0}    # Costo dal nodo start al nodo corrente
    f_score = {start: heuristic(start, goal,rows,cols)}  # f = g + h
    
    # Variabile per tenere traccia del nodo raggiunto con il massimo numero di passi (g)
    best_candidate = start
    best_g = g_score[start]

    while open_set:
        _, current = heapq.heappop(open_set)
        
        # Aggiorna best_candidate se abbiamo raggiunto un nodo con g maggiore
        if g_score[current] > best_g:
            best_g = g_score[current]
            best_candidate = current

        # Se raggiungiamo l'obiettivo, ricostruiamo e restituiamo il percorso completo.
        if current ==tuple(goal) :
            
            return reconstruct_path(came_from, current)
        
        # Se il costo per arrivare qui ha raggiunto il limite massimo, non espandiamo ulteriormente.
        if g_score[current] >= max_depth:
            continue
        
        # Esploriamo i vicini (su, giù, sinistra, destra)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = ((current[0] + dx) % rows , (current[1] + dy) % cols)
            

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
                    f = tentative_g + heuristic(neighbor, goal,rows,cols)
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

