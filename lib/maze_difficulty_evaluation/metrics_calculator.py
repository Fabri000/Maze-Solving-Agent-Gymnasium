from lib.a_star_algos.a_star import astar_limited_partial

class MetricsCalculator:
    """
    Class that calculate the metrics for a maze that will be used to evaluate its difficulty
    following the approach of the paper "The Quest for the Perfect Perfect-Maze" by Kim and Crawfis.
    Args:
        maze (array): maze representation.
        sol_path_length (int): length of the solution path.
    """
    def __init__(self, maze: list[list[int]],sol_path_length:int):
        self.maze = maze
        self.sol_path_length = sol_path_length
        self.maze_size = (len(maze), len(maze[0]))
        self.goal = [(r, c) for r in range(self.maze_size[0]) for c in range( self.maze_size[1]) if  self.maze[r][c] == 2][0]
        self.CE = (self.maze_size[0]-1) * (( self.maze_size[1]-1)//2) - 1

    def calculate_density(self):
        "It calculates the density of the maze w.r.t the number of walkable cells in relation to the size of the maze."
        return sum(1 for r in range(self.maze_size[0]) for c in range(self.maze_size[1]) if self.maze[r][c] != 0) / (self.maze_size[0] * self.maze_size[1])

    def calculate_L(self,path):
        """It calculates the lenght of a path in relation to the number of cells in the maze
        Args:
            path (list): the path to evaluate."""
        return  len(path)/self.CE
    
    def calculate_T(self,path):
        """It calculates the number of turns in a path in relation to the number of cells in the  solution path
        Args:
            path (list): the path to evaluate."""
        turns = 0
        for i in range(1,len(path)-1):
            if path[i-1][0] != path[i+1][0] and path[i-1][1] != path[i+1][1]:
                turns += 1
            
        return turns/self.sol_path_length

    def calculate_J(self,path):
        """It calculates the number of junctions in a path in relation to the number of cells in the solution path
        Args:
            path (list): the path to evaluate.
        Returns:
            float: the percentage of junctions in the path."""
        junctions = 0
        for i in range(len(path)):
            pos = path[i]
            
            neighbors =  sum(1 for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]if self.maze[pos[0] + dx][pos[1] + dy] != 0)
            if neighbors == 3:
                junctions += 1

        return junctions/self.sol_path_length
    
    def calculate_CR(self,path):
        """It calculates the number of cross roads in a path in relation to the number of cells in the solution path
        Args:
            path (list): the path to evaluate.
        Returns:
            float: the percentage of cross roads in the path."""
        cross = 0
        for i in range(len(path)):
            pos = path[i]
            
            neighbors =  sum(1 for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]if self.maze[pos[0] + dx][pos[1] + dy] != 0)
            if neighbors == 4:
                cross += 1

        return cross/self.sol_path_length
    
    def calculate_D(self,path):
        """It calculates the number of decision point in a path in relation to the number of cells in the solution path.
        Args:
            path (list): the path to evaluate.
        Returns:
            float: the percentage of decision point in the path."""
        
        decisions = 0
        for i in range(len(path)):
            pos = path[i]
            
            neighbors =  sum(1 for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]if self.maze[pos[0] + dx][pos[1] + dy] != 0)
            if neighbors > 2:
                decisions += 1
        return decisions/self.sol_path_length

    def calculate_DE(self,path):
        """It calculates the number of dead ends into the maze in relation to the number of cells in the solution path
        Args:
            path (list): the path to evaluate.
        Returns:
            float: the percentage of dead ends point in the path."""
        
        AC,FDE,BDE = self.calculate_DE_sub(path)
        return AC+FDE+BDE

    def calculate_DE_sub(self,path):
        """It calculates Alcoves, Forward and Backward dead ends in relation to the number of cells in the solution path.
        Args:
            path (list): the path to evaluate."""
        
        de_points = self.extract_de_points(path)

        alcoves = 0
        forward = 0
        backward = 0

        decision_points = []
        for point in de_points:
            de_path = self.calculate_path(point,path)
            if len(set(de_path) & set(decision_points)) == 0 :
                for k in range(1,len(de_path)-1):
                    i,j = de_path[k]
                    neighbors =  sum(1 for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]if self.maze[i + dx][j + dy] != 0)
                    if neighbors > 2:
                        decision_points.append((i,j))
                        break

                type= self.type_of_DE(de_path)
                if type == "FDE":
                    forward += 1
                elif type == "BDE":
                    backward += 1
                else:
                    alcoves+=1

        return alcoves/self.sol_path_length,forward/self.sol_path_length,backward/self.sol_path_length
    
    def extract_de_points(self, path):
        """It extracts the dead end points from the maze."""
        de_points = []
        for i in range(1,len(self.maze)-1):
            for j in range(1,len(self.maze[0])-1):
                if self.maze[i][j] == 1:
                    neighbors =  sum(1 for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]if self.maze[i + dx][j + dy] != 0)
                    if neighbors == 1 and (i,j) not in path:
                        de_points.append((i,j))
        return de_points

    def  calculate_path(self,point,path): 
        """It calculates the path from a point to a decision cell into the path solution.
        Args:
            point (tuple): the starting point.
            path (list): the path to evaluate."""
        
        de_path = astar_limited_partial(self.maze,point,path[0])
        for i in range(1,len(path)-1): # probabile problema qui vedi maze_complexity_evaluation
            if de_path[i] in path:
                de_path = de_path[:i]
                break
        return de_path
    
    def type_of_DE(self,path):
        """Return type of dead end: AC for alcove, FDE for forward dead end and BDE for backward dead end"""
        flag = False
        for k in range(1,len(path)-1):
                i,j = path[k]
                neighbors =  sum(1 for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]if self.maze[i + dx][j + dy] != 0)
                if neighbors >2:
                    flag = True
                    break
                if self.calculate_T(path) > 0:
                    flag = True
                    break

        if flag:
            diff_dist = (abs(path[-1][0]-self.goal[0])+abs(path[-1][1]-self.goal[1])) - (abs(path[0][0]-self.goal[0])+abs(path[0][1]-self.goal[1]))
            if diff_dist > 0:
                return "FDE"
            else:
                return "BDE"
        else:
            return "AC"

    def calculate_T_DE(self,path,type):
        """
        It calculates the number of turning points for specified type of dead ends cell rooted in solution path.
        """
        de_points =self.extract_de_points(path)
        sum = 0
        for point in de_points:
            de_path = self.calculate_path(point,path)
            if self.type_of_DE(de_path) == type:
                sum += self.calculate_T(de_path) / len(de_path)
        return sum 
        
    def calculate_D_sharp(self,path,type):
        """
        It calculates the number of decision points for specified type of dead ends rooted in solution path.
        """
        de_points =self.extract_de_points(path)
        sum = 0
        for point in de_points:
            de_path = self.calculate_path(point,path)
            if self.type_of_DE(de_path) == type:
                sum += self.calculate_D(de_path) / len(de_path)
        return sum
    
    def calculate_L_sharp(self,path,type):
        """
        It calculates the lenght for specified type of dead ends rooted in solution path.
        """
        de_points =self.extract_de_points(path)
        d_points = []
        sum = 0
        for point in de_points:
            de_path = self.calculate_path(point,path)
            if self.type_of_DE(de_path) == type:
                if len(set(d_points) & set(de_path))==0:
                        d = self.find_decision(de_path)
                        if  d:
                            d_points.append(d)
                        sum +=  len(de_path)/self.CE
                else:   
                        de_path.reverse()
                        d = self.find_decision(de_path)
                        if  d:
                            de_path = de_path[0:de_path.index(d)]
                            d = self.find_decision(de_path)
                            if d:
                                de_path = de_path[0:de_path.index(d)]
                        sum +=  len(de_path)/self.CE
        return sum

    def calculate_L_DE(self,path):
        de_points =self.extract_de_points(path)
        sum = 0
        d_points = []
        for point in de_points:
            de_path = self.calculate_path(point,path)
            if len(set(d_points) & set(de_path))==0:
                d = self.find_decision(de_path)
                if  d:
                    d_points.append(d)
                sum +=  len(de_path)/self.CE
            else:
                d = self.find_decision(de_path)
                if  d:
                    de_path = de_path[0:de_path.index(d)]
                    d = self.find_decision(de_path)
                    if d:
                        de_path = de_path[0:de_path.index(d)]
                sum +=  len(de_path)/self.CE
        return sum
    
    def find_decision(self,path):
        """Find the closest decision point to the point on the solution path
        Args:
            path:array:the path"""
        for k in range(1,len(path)-1,-1):
                i,j = path[k]
                neighbors =  sum(1 for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]if self.maze[i + dx][j + dy] != 0)
                if neighbors > 2:
                    return (i,j)
        return None
    


