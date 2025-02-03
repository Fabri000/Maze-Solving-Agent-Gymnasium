import pygame
import numpy as np

class MazeView():
    DIR = {
        'N': (-1,0),
        'S': (1,0),
        'E': (0,-1),
        'W': (0,1)
    }

    TILE_SIZE = 16
    CELL_COLORS = [(0,0,0),(255,255,255),(3, 255, 28)] #colors of wall, floor, goal
    AGENT_COLOR = (17, 0, 255)


    def __init__(self, maze_map, start_position, goal_position, maze_size: tuple[int,int], enable_render:bool=True):
        
        self.game_over = False
        self.enable_render = enable_render
        
        self.maze_size = maze_size
        self.window_size = tuple([x*MazeView.TILE_SIZE for x in reversed(self.maze_size)])
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
            self.__draw_agent()

            self.screen.blit(self.background,(0,0))
            self.screen.blit(self.maze_layer,(0,0))
            pygame.display.flip()
            pygame.display.update()

    
    def __draw_maze(self):
        if not self.enable_render:
            return
    
        for i in range(self.maze_size[0]):
            for j in range(self.maze_size[1]):
                x,y = j * MazeView.TILE_SIZE, i * MazeView.TILE_SIZE
                color = MazeView.CELL_COLORS[self.maze_map[i][j]]
                pygame.draw.rect(self.maze_layer,color,pygame.Rect(x,y,MazeView.TILE_SIZE,MazeView.TILE_SIZE),0)
                pygame.draw.rect(self.maze_layer,(148, 148, 148),pygame.Rect(x,y,MazeView.TILE_SIZE,MazeView.TILE_SIZE),1)
        
    def __draw_agent(self,transparency=255):
        if not self.enable_render:
            return
        x,y = self._agent_position[1] * MazeView.TILE_SIZE + MazeView.TILE_SIZE // 4 , self._agent_position[0] * MazeView.TILE_SIZE +  MazeView.TILE_SIZE // 4

        pygame.draw.rect(self.maze_layer,MazeView.AGENT_COLOR+(transparency,),pygame.Rect(x,y,MazeView.TILE_SIZE // 2,MazeView.TILE_SIZE // 2),0)

    def move_agent(self, dir):
        new_pos = self._agent_position + np.array(dir)
        if 0 < new_pos[0]< len(self.maze_map)-1 and  0 < new_pos[1] < len(self.maze_map[0])-1:
            if self.maze_map[new_pos[0]][new_pos[1]] != 0:
                # update the drawing
                self.__draw_agent(transparency=0)
                # redraw previous cell
                self.__draw_cell(self._agent_position)
                # move the robot
                self._agent_position += np.array(dir)
                # if it's in a portal afterward
                self.__draw_agent(transparency=255)
                self.update()
                return True
        
        return False

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
            self.__draw_agent()
            
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
    
    def _reset_agent(self):
        self.__draw_agent(transparency=0)
        self.__draw_cell(self._agent_position)
        self._agent_position = self.start_position
        self.__draw_agent()

    def quit_game(self):
        try:
            self.game_over = True
            if self.__enable_render:
                pygame.display.quit()
            pygame.quit()
        except Exception:
            pass

    def __draw_cell(self,position):
        color = MazeView.CELL_COLORS[self.maze_map[position[0]][position[1]]]
        x,y = position[1] * MazeView.TILE_SIZE, position[0] * MazeView.TILE_SIZE
        pygame.draw.rect(self.maze_layer,color,pygame.Rect(x,y,MazeView.TILE_SIZE,MazeView.TILE_SIZE),0)
        pygame.draw.rect(self.maze_layer,(148, 148, 148),pygame.Rect(x,y,MazeView.TILE_SIZE,MazeView.TILE_SIZE),1)