import pygame
from lib.maze_view import MazeViewTemplate
import numpy as np

maze = [[0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0], [0, 1, 0, 1, 0, 0, 0], [0, 1, 0, 1, 1, 1, 0], [0, 0, 0, 1, 0, 0, 0], [0, 1, 1, 1, 2, 1, 0], [0, 0, 0, 0, 0, 0, 0]]
view = MazeViewTemplate(maze,(1,1),(1,1),maze_size=(len(maze),len(maze[0])))

ACTIONS = {           
            0: np.array([1, 0]),  # down
            1: np.array([0, 1]),  # right
            2: np.array([-1, 0]),  # up
            3: np.array([0, -1]),  # left
        }


while True:
    pygame.time.delay(100) # This will delay the game the given amount of milliseconds. In our casee 0.1 seconds will be the delay

    for event in pygame.event.get():  # This will loop through a list of any keyboard or mouse events.
        if event.type == pygame.QUIT: # Checks if the red button in the corner of the window is clicked
            run = False  # Ends the game loop
        keys = pygame.key.get_pressed()  # This will give us a dictonary where each key has a value of 1 or 0. Where 1 is pressed and 0 is not pressed.

        if keys[pygame.K_LEFT]: # We can check if a key is pressed like this
            view.move_agent(ACTIONS[3])
        if keys[pygame.K_RIGHT]:
            view.move_agent(ACTIONS[1])
        if keys[pygame.K_UP]:
            view.move_agent(ACTIONS[2])
        if keys[pygame.K_DOWN]:
            view.move_agent(ACTIONS[0])
        if keys[pygame.K_q]:
            pygame.quit()
        view.view_update()