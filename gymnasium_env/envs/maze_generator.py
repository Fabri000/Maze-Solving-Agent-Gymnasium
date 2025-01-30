import random

def gen_maze(raws: int, columns: int) -> list:
    maze = [[0 for _ in range(columns)] for _ in range(raws)]

    start_point = (random.randint(1,len(maze)-2),random.randint(1,len(maze[0])-2))
    simulate_visit(maze, start_point, raws, columns)

    return start_point,maze

def simulate_visit(maze, start_point, r, c):
    stack = [start_point]
    
    while stack:
        x, y = stack.pop()
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        random.shuffle(directions)
        
        for dx, dy in directions:
            nx, ny = x + 2 * dx, y + 2 * dy
            if 0 < nx < r - 1 and 0 < ny < c - 1 and maze[nx][ny] == 0:
                maze[x + dx][y + dy] = 1
                maze[nx][ny] = 1
                stack.append((nx, ny))
    
    pos = find_random_position(maze,1,start_point)
    maze[pos[0]][pos[1]] = 2
    

def find_random_position(maze, val, start_point):
    # Trova tutte le posizioni che contengono il valore specificato
    positions = [(r, c) for r in range(len(maze)) for c in range(len(maze[0])) if maze[r][c] == val]
    positions.remove(start_point)

    if not positions:
        return None
    
    chosed = random.choice(positions)
    while abs(chosed[0]-start_point[0])+abs(chosed[1]-start_point[1]) < max(len(maze),len(maze[0])) // 2:
        chosed = random.choice(positions)

    return chosed