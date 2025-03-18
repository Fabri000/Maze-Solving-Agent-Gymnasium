import math
import networkx as nx
import matplotlib.pyplot as plt
from lib.a_star_algos.a_star import astar_limited_partial


def cantor_pairing(point: tuple[int, int]) -> int:
        """
        Applica la funzione di accoppiamento di Cantor su una coppia di numeri per ottenere un ID univoco.
        
        Formula: F(x, y) = (x + y) * (x + y + 1) // 2 + y

        Args:
            point (tuple[int, int]): La coppia di coordinate (x, y).

        Returns:
            int: L'ID univoco generato.
        """
        x, y = point
        return (x + y) * (x + y + 1) // 2 + y

    
def inverse_cantor_pairing(id:int):
        """It converts the unique id obtained using cantor pairy function
         Args:
            id (int): the unique id
        Returns:
            tuple[int,int]: the point
        """

        w = math.floor( (math.sqrt(8*id + 1) - 1) / 2)
        t = (w**2 + w) // 2
        y = id - t
        x = w - y
        return (x,y)


class ComplexityEvaluation:
    """Class that evaluate the complexity and difficulty of a maze w.r.t method described into 'The complexity and difficulty of maze' by McCledon"""

    def __init__(self,maze,start_pos,goal_pos):
        self.start_pos = start_pos
        self.maze = maze
        self.goal_pos = goal_pos
        self.G = nx.Graph()
        self.init_graph_repr(maze,start_pos,goal_pos)

        """labels = {}
        for n in self.G.nodes:
            labels[n] = inverse_cantor_pairing(n)
        nx.draw(self.G,labels,labels=labels,with_labels = True)
        edge_labels = nx.get_edge_attributes(self.G,"d")
        nx.draw_networkx_edge_labels(self.G,labels, edge_labels = edge_labels)
        plt.savefig('maze.png')"""

    
    def init_graph_repr(self,maze,start_pos,goal_pos):
        """
        Generate a graph representing the maze using the process described into 'The complexity and difficulty of maze' by McCledon
        """
        
        solution = astar_limited_partial(maze,start_pos,goal_pos)
        s_nodes = self.decompose_in_turns(maze,solution)

        self.create_graph_branch(s_nodes)
        solution_branch = self.G.copy()

        d_h  = {}
        d_h.update(self.calculate_lenght_arcs(s_nodes,solution)) 

        nx.set_edge_attributes(solution_branch,d_h,"d")

        junctions = self.get_junctions(s_nodes)
        dead_ends = self.get_dead_ends(solution)
        for dead_end in dead_ends:
            path = self.calculate_path(dead_end)
            p_nodes = self.decompose_in_turns(maze,path)
            junctions += self.get_junctions(p_nodes)
            self.create_graph_branch(p_nodes)
            d_h.update(self.calculate_lenght_arcs(p_nodes,path))

        nx.set_edge_attributes(self.G,d_h,"d")
        junctions = list(set(junctions))

        subs = self.extract_hallways(junctions,s_nodes)
        self.hallways = {0: solution_branch}
        for i, subgraph in enumerate(subs):
            self.hallways[i+1]=subgraph
        
        self.branches = self.get_branches(junctions,s_nodes)
        self.branches[0]= [0]

    def tmp(self,subs):
        g = nx.Graph()

        for _,subgraph in enumerate(subs):
            g.add_nodes_from(subgraph.nodes)
            g.add_edges_from(subgraph.edges)

        labels = {}
        for n in g.nodes:
            labels[n] = inverse_cantor_pairing(n)
        nx.draw(g,labels,labels=labels,with_labels = True)
        edge_labels = nx.get_edge_attributes(subgraph,"d")
        nx.draw_networkx_edge_labels(g,labels, edge_labels = edge_labels)
        plt.show()
        
    def hallways_info(self):
        for i, subgraph in enumerate(self.hallways.values()):
            print(f"Subgraph {i+1}:")
            print("  Nodes:", [inverse_cantor_pairing(n) for n in subgraph.nodes()])
            print("  Edges:", subgraph.edges(data=True))
            print("-" * 20)

    def create_graph_branch(self,nodes):
        self.G.add_node(cantor_pairing(nodes[0]))
        for i in range(1,len(nodes)-1):
            id = cantor_pairing(nodes[i])
            self.G.add_node(id)
            self.G.add_edge(cantor_pairing(nodes[i-1]),id)
        
        self.G.add_node(cantor_pairing(nodes[-1]))
        self.G.add_edge(cantor_pairing(nodes[-2]),cantor_pairing(nodes[-1]))

    def decompose_in_turns(self,maze,path):
        """
        Returns the sequence of dead ends and turn into the path.
        """
        ris = [path[0]]
        for i in range(1,len(path)-1):
            pos = path[i]
            neighbors =  sum(1 for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]if maze[pos[0] + dx][pos[1] + dy] != 0)
            if path[i-1][0] != path[i+1][0] and path[i-1][1] != path[i+1][1] or neighbors > 2:
                ris.append(path[i])
        ris.append(path[-1])
        return ris
    
    def get_junctions(self,path):
        """Get the junctions into a  path
        Args:
            path (list)
        Returns:
            list: list of junctions in path"""
        js = []
        for i in range(len(path)):
            pos = path[i]
            neighbors =  sum(1 for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]if self.maze[pos[0] + dx][pos[1] + dy] != 0)
            if neighbors == 3:
                js.append(path[i])
        return js
    
    def get_dead_ends(self,path):
        """Get the dead end into the maze
        Args:
            path (list)
        Returns:
            list: list of dead ends"""
        
        de_points = []
        for i in range(1,len(self.maze)-1):
            for j in range(1,len(self.maze[0])-1):
                if self.maze[i][j] == 1:
                    neighbors =  sum(1 for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]if self.maze[i + dx][j + dy] != 0)
                    if neighbors == 1 and (i,j) not in path:
                        de_points.append((i,j))
        return de_points
    
    def  calculate_path(self,point):
        """It calculates the path from a point to a junction cell into the path solution.
        Args:
            point (tuple): the starting point.
            path (list): the path to evaluate."""
        
        return astar_limited_partial(self.maze,point,self.start_pos)
    
    def calculate_lenght_arcs(self,ns,path):
        """Calculate the  lenght of each arch into the branch"""
        values = {}
        for i in range(len(ns)-1):
            id1 = cantor_pairing(ns[i])
            id2 = cantor_pairing(ns[i+1])
            values[(id1,id2)]= len(path[path.index(ns[i]):path.index(ns[i+1])-1])
        
        return values

    def extract_hallways(self, split_points, solution_points):
        """
        Extracts connected components from a graph, including split points and their edges.
        Args:
            split_points (list[tuple]): junction pointh on wich branches originates.
            solution_points (list[tuple]): edge in graph associated to solution.
        """

        temp_graph = self.G.copy()

        p = [cantor_pairing(n) for n in split_points]
        temp_graph.remove_nodes_from(p)
        s_p = [cantor_pairing(x) for x in solution_points]
        temp_graph.remove_nodes_from(s_p)

        base_components = list(nx.connected_components(temp_graph))
        result_subgraphs = []

        for component_nodes in base_components:
            component_nodes = set(component_nodes)

            # Find adjacent split points (only those directly connected to component_nodes)
            adjacent_split_points = set()
            for node in component_nodes:
                for neighbor in self.G.neighbors(node):
                    if neighbor in p:
                        adjacent_split_points.add(neighbor)
                        if neighbor in s_p:
                            break

            # Create a subgraph that includes the base component and its direct split point neighbors.
            all_nodes = component_nodes.union(adjacent_split_points)
            final_subgraph = self.G.subgraph(all_nodes)
            result_subgraphs.append(final_subgraph)

        return result_subgraphs

    def get_branches(self,split_points,solution_points):
        """
        Get the brances of the maze, and the associated hallways
        Args:
            split_points (list): junction points.
            solution_points(list): point into the solution.
        Returns:
            Dict: association beetween branch and hallways that compose it.
        """
        temp_graph = self.G.copy()

        s_p = [cantor_pairing(x) for x in solution_points if x not in split_points]
        temp_graph.remove_nodes_from(s_p)

        base_components = list(nx.connected_components(temp_graph))

        branches = []
        for component_nodes in base_components:
            component_nodes = set(component_nodes)

            final_subgraph = self.G.subgraph(component_nodes)
            branches.append(final_subgraph)
        
        hallways = self.hallways.copy()

        branch_comp = {}
        b = 1
        for branch in branches:
            branch_comp[b] = []
            keys = list(hallways.keys())
            for i in keys:
                h = hallways[i]
                if set(h.nodes) <= set(branch.nodes):
                    hallways.pop(i)
                    branch_comp[b].append(i)
            b += 1
        return branch_comp
    
    def number_of_turns(self,h):
        """
        Calculate the number of turns into the h-th hallway.
        Args:
            h (int): id of hallways
        """
        hallway = self.hallways[h]
        turns = 0
        for n in hallway.nodes:
            point = inverse_cantor_pairing(n)
            neighbors =  sum(1 for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]if self.maze[point[0] + dx][point[1] + dy] != 0)
            if neighbors > 1 :
                turns+=1
        return turns
    
    def total_dimension_arc(self,h):
        """
        Calculate the total dimension of turns into the h-th hallway.
        Args:
            h (int): id of the hallway
        """
        hallway = self.hallways[h]
        ds = nx.get_edge_attributes(hallway,"d")
        return sum([ds[e] for e in ds.keys()])
    
    def complexity_of_hallway(self,h):
        """It calculates the complexity of the h-th hallway.
        Args:
            h (int): id of the hallway
        """
        D_h = self.total_dimension_arc(h)
        hallway = self.hallways[h]
        ds = nx.get_edge_attributes(hallway,"d")
        
        s = sum([1/(2*ds[e]) for e in ds.keys()])
        return D_h * s

    def complexity_of_branch(self,i):
        """
        Calculate the complexity of the branch.
        Args:
            i (int):id of the branch.
        """
        s = 0
        hallways = self.branches[i]
        for hallway in hallways:
            s+=self.complexity_of_hallway(hallway)
        return s
        
    def complexity_of_maze(self):
        """
        It calculates the total complexity of the maze.
        """
        s = 0
        for b in self.branches:
            s += self.complexity_of_branch(b)
        return math.log(s)
    
    def difficulty_of_maze(self):
        """
        It calculates the difficulty of the maze.
        """
        p = 1
        for h in self.branches.keys():
            if h == 0:
                p*=self.complexity_of_branch(0)
            else:
                p*= self.complexity_of_branch(h)+1
        return math.log(p)
    
    def show_branch_info(subs):
        """
        Draw branches and give info about sub branches of graph.
        """
        for i,subgraph in enumerate(subs):
            labels = {}
            for n in subgraph.nodes:
                labels[n] = inverse_cantor_pairing(n)
            nx.draw(subgraph,labels,labels=labels,with_labels = True)
            edge_labels = nx.get_edge_attributes(subgraph,"d")
            nx.draw_networkx_edge_labels(subgraph,labels, edge_labels = edge_labels)
            plt.show()
