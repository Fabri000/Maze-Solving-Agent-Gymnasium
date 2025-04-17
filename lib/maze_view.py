import pygame
import numpy as np

class MazeViewTemplate():
    DIR = {
        'N': (-1,0),
        'S': (1,0),
        'E': (0,-1),
        'W': (0,1)
    }

    TILE_SIZE = 16
    CELL_COLORS = [(46, 52, 64),(236, 239, 244),(163, 190, 140)] #colors of wall, floor, goal
    AGENT_COLOR = (94, 129, 172)

    def __init__(self, maze_map, start_position, goal_position, maze_size: tuple[int,int], enable_render:bool=True):
        
        self.game_over = False
        self.enable_render = enable_render
        
        self.maze_size = maze_size
        self.window_size = tuple([x*MazeViewTemplate.TILE_SIZE for x in reversed(self.maze_size)])
        self.maze_map = maze_map

        self.start_position = start_position
        self.goal_position = goal_position

        self._agent_position = self.start_position

        self.screen = None
        self.clock = None

        if self.enable_render:
            pygame.init()
            pygame.display.set_caption("Maze game")
            pygame.display.init()

            # to show the right and bottom border
            self.screen = pygame.display.set_mode(self.window_size)
            self.__screen_size = tuple(map(sum, zip(self.window_size, (-1, -1))))
            self.background = pygame.Surface(self.__screen_size).convert()

            self.maze_layer = pygame.Surface(self.__screen_size).convert_alpha()
            # show the maze
            self.__draw_maze()

            # show the robot
            self._draw_agent()

            self.screen.blit(self.background,(0,0))
            self.screen.blit(self.maze_layer,(0,0))
            pygame.display.flip()
            pygame.display.update()
    
    def update_maze(self,maze_map, start_position, goal_position, maze_size):
        self.maze_map = maze_map
        self.start_position = start_position
        self._agent_position = self.start_position
        self.goal_position = goal_position
        self.maze_size = maze_size

        self.window_size = tuple([x*MazeViewTemplate.TILE_SIZE for x in reversed(self.maze_size)])
        
        if self.enable_render:
            pygame.init()
            pygame.display.set_caption("Maze game")
            pygame.display.init()

            # to show the right and bottom border
            self.screen = pygame.display.set_mode(self.window_size)
            self.__screen_size = tuple(map(sum, zip(self.window_size, (-1, -1))))
            self.background = pygame.Surface(self.__screen_size).convert()

            self.maze_layer = pygame.Surface(self.__screen_size).convert_alpha()
            # show the maze
            self.__draw_maze()

            # show the robot
            self._draw_agent()

            self.screen.blit(self.background,(0,0))
            self.screen.blit(self.maze_layer,(0,0))
            pygame.display.flip()
            pygame.display.update()
    
    def __draw_maze(self):
        if not self.enable_render:
            return
        for i in range(self.maze_size[0]):
            for j in range(self.maze_size[1]):
                x,y = j * MazeViewTemplate.TILE_SIZE, i * MazeViewTemplate.TILE_SIZE
                color = MazeViewTemplate.CELL_COLORS[self.maze_map[i][j]]
                pygame.draw.rect(self.maze_layer,color,pygame.Rect(x,y,MazeViewTemplate.TILE_SIZE,MazeViewTemplate.TILE_SIZE),0)
                pygame.draw.rect(self.maze_layer,(59, 66, 82),pygame.Rect(x,y,MazeViewTemplate.TILE_SIZE,MazeViewTemplate.TILE_SIZE),1)
        
    def _draw_agent(self,transparency=255):
        if not self.enable_render:
            return
        x,y = self._agent_position[1] * MazeViewTemplate.TILE_SIZE + MazeViewTemplate.TILE_SIZE // 4 , self._agent_position[0] * MazeViewTemplate.TILE_SIZE +  MazeViewTemplate.TILE_SIZE // 4

        pygame.draw.rect(self.maze_layer,MazeViewTemplate.AGENT_COLOR+(transparency,),pygame.Rect(x,y,MazeViewTemplate.TILE_SIZE // 2,MazeViewTemplate.TILE_SIZE // 2),0)

    def move_agent(self, dir):
        '''
        Move the agent in the maze following the direction given.
        Args:
            dir (tuple): direction to move
        '''
        pass

    def update(self, mode="human"):
        try:
            img_output = self.view_update(mode)
            self.__controller_update()
        except Exception as e:
            self.game_over = True
            self.quit_game()
            raise e
        else:
            return img_output
        
    def view_update(self, mode="human"):
        if not self.game_over:
            # update the robot's position
            self._draw_agent()
            
            self.screen.blit(self.background,(0,0))
            self.screen.blit(self.maze_layer,(0,0))

            if mode == "human":
                # update the screen
                pygame.display.update()

            return np.flipud(np.rot90(pygame.surfarray.array3d(pygame.display.get_surface())))

    def __controller_update(self):
        if not self.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.game_over = True
                    self.quit_game()
    
    def _draw_cell(self,position):
        color = MazeViewTemplate.CELL_COLORS[self.maze_map[position[0]][position[1]]]
        x,y = position[1] * MazeViewTemplate.TILE_SIZE, position[0] * MazeViewTemplate.TILE_SIZE
        pygame.draw.rect(self.maze_layer,color,pygame.Rect(x,y,MazeViewTemplate.TILE_SIZE,MazeViewTemplate.TILE_SIZE),0)
        pygame.draw.rect(self.maze_layer,(208, 135, 112),pygame.Rect(x,y,MazeViewTemplate.TILE_SIZE,MazeViewTemplate.TILE_SIZE),1) #(59, 66, 82)

    def _reset_agent(self):
        self._draw_agent(transparency=0)
        self._draw_cell(self._agent_position)
        self._agent_position = self.start_position
        self._draw_agent()

    def quit_game(self):
        try:
            self.game_over = True
            if self.__enable_render:
                pygame.display.quit()
            pygame.quit()
        except Exception:
            pass


class SimpleMazeView(MazeViewTemplate):

    def move_agent(self, dir):
        new_pos = self._agent_position + np.array(dir)
        if 0 < new_pos[0]< len(self.maze_map)-1 and  0 < new_pos[1] < len(self.maze_map[0])-1:
            if self.maze_map[new_pos[0]][new_pos[1]] != 0:
                self._draw_agent(transparency=0)
                self._draw_cell(self._agent_position)

                self._agent_position += np.array(dir)

                self._draw_agent(transparency=255)
                self.update()
                return True
        
        return False

class ToroidalMazeView(MazeViewTemplate):

    def move_agent(self, dir):
        new_pos = self._agent_position + np.array(dir)
        new_pos = (new_pos[0] % self.maze_size[0], new_pos[1] % self.maze_size[1])
        if self.maze_map[new_pos[0]][new_pos[1]] != 0:
            self._draw_agent(transparency=0)
            self._draw_cell(self._agent_position)

            self._agent_position = new_pos

            self._draw_agent(transparency=255)
            self.update()
            return True
        
        return False
    