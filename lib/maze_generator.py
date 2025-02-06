import random

def gen_maze(shape:tuple[int,int], algorithm:str="dfs") -> list:
    rows,columns = shape
    maze = [[0 for _ in range(rows)] for _ in range(columns)]
    
    start_point= (random.randrange(1, rows - 1, 2), random.randrange(1, columns - 1, 2))
    maze[start_point[0]][start_point[1]] = 1

    match algorithm:
        case "r-prim":
            random_prim_visit(maze,rows,columns,start_point)
        case "dfs":
            deept_first_visit(maze,rows,columns,start_point)

    win_pos = find_random_position(maze,1,start_point)
    maze[win_pos[0]][win_pos[1]] = 2

    return start_point , maze

def random_prim_visit(maze,width, height,start_point):
    
    # Lista delle celle frontiera
    frontier = []
    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        nx, ny = start_point[0] + dx, start_point[1] + dy
        if 0 < nx < height - 1 and 0 < ny < width - 1:  # Evita di uscire dai bordi
            frontier.append((nx, ny))
    
    while frontier:
        # Scegli una cella frontiera casuale
        fx, fy = random.choice(frontier)
        frontier.remove((fx, fy))
        
        # Trova tutte le celle adiacenti che sono già parte del labirinto
        neighbors = []
        for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            nx, ny = fx + dx, fy + dy
            if 0 <= nx < height and 0 <= ny < width and maze[nx][ny] == 1:
                neighbors.append((nx, ny))
        
        if neighbors:
            # Scegli una cella adiacente casuale
            nx, ny = random.choice(neighbors)
            
            # Collega la cella frontiera alla cella adiacente
            maze[fx][fy] = 1
            maze[(fx + nx) // 2][(fy + ny) // 2] = 1  # Rimuovi il muro tra le due celle
            
            # Aggiungi le nuove celle frontiera
            for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                nx, ny = fx + dx, fy + dy
                if 0 < nx < height - 1 and 0 < ny < width - 1 and maze[nx][ny] == 0:
                    frontier.append((nx, ny))



def deept_first_visit(maze, width, height, start_point):
    stack = [start_point]
    
    while stack:
        x, y = stack.pop()
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        random.shuffle(directions)
        
        for dx, dy in directions:
            nx, ny = x + 2 * dx, y + 2 * dy
            if 0 < nx < height - 1 and 0 < ny < width - 1 and maze[nx][ny] == 0:
                maze[x + dx][y + dy] = 1
                maze[nx][ny] = 1
                stack.append((nx, ny))


def find_random_position(maze, val, start_point):
    # Trova tutte le posizioni che contengono il valore specificato
    positions = [(r, c) for r in range(len(maze)) for c in range(len(maze[0])) if maze[r][c] == val]
    positions.remove(start_point)

    if not positions:
        return None
    
    chosed = random.choice(positions)
    while abs(chosed[0]-start_point[0])+abs(chosed[1]-start_point[1]) < max(len(maze),len(maze[0])) - max(len(maze),len(maze[0]))/2:
        chosed = random.choice(positions)

    return chosed