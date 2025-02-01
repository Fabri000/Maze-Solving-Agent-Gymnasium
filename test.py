from lib.a_star import astar_limited_partial


# --- Simulazione con i dati forniti ---
if __name__ == "__main__":
    start = (2, 1)  # (riga, colonna)
    
    maze = [
        [0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,1,0,1,0],
        [0,1,0,0,0,0,0,1,0,1,0],
        [0,1,0,2,0,1,0,1,0,1,0],
        [0,1,0,1,1,1,1,1,0,1,0],
        [0,1,0,1,0,0,0,1,1,1,0],
        [0,1,0,0,0,1,0,0,0,1,0],
        [0,1,1,1,1,1,1,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0]
    ]
    
    win_pos = (3, 3)  # L'obiettivo si trova in (5,3)
    max_depth = 6     # Limite massimo di passi consentiti
    
    path = astar_limited_partial(maze, start, goal=win_pos, max_depth=max_depth)
    
    print("Percorso ottenuto:")
    for step in path:
        print(step)