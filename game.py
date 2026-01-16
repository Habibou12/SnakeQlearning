import math
import sys
import random

import pygame
import numpy as np
from Trainer import *

from utils import *

pygame.init()
pygame.display.set_caption("2048 Game")

class Game:
    def __init__(self, affiche):
        self.affiche = affiche
        if affiche:
            self.screen = pygame.display.set_mode((600,600))
            self.display = pygame.Surface((16*40,16*40))
            self.clock = pygame.time.Clock()
        self.snake = []
        self.snake.append([10,5])
        for i in range(4):
            self.snake.append([self.snake[-1][0] - 1, self.snake[-1][1]])

        self.apple = []
        self.spawnApple()
        self.dir = 1
        self.action = [(1,0),(0,-1),(-1,0),(0,1)]
    def reset(self):
        self.spawnApple()
        self.snake = []
        self.snake.append([10, 5])
        for i in range(4):
            self.snake.append([self.snake[-1][0] - 1, self.snake[-1][1]])
        self.dir = 1

    def spawnApple(self):
        self.apple = [np.random.randint(0, 39), np.random.randint(0, 39)]
        while self.apple in self.snake:
            self.apple = [np.random.randint(0, 39), np.random.randint(0, 39)]
    def is_collision(self, point):

        if point in self.snake or point[0] < 0 or point[0] >= 40 or point[1] < 0 or point[1] >= 40:
            return True
        return False

    def step(self, dir):
            self.dir = dir

            for i in range(len(self.snake)-1, 0, -1):
                self.snake[i][0] = self.snake[i-1][0]
                self.snake[i][1] = self.snake[i - 1][1]


            death = False
            reward = 0
            action = self.action[self.dir-1]
            self.snake[0][0] += action[0]
            self.snake[0][1] += action[1]
            if self.apple in self.snake:
                reward = 20
                self.spawnApple()
                self.snake.append([self.snake[-1][0] - action[0],self.snake[-1][1] - action[1]])
            if self.snake[0] in self.snake[1:] or self.snake[0][0] < 0 or self.snake[0][0] >= 40 or self.snake[0][1] < 0 or self.snake[0][1] >= 40:
                reward = -20
                death = True
            if self.affiche:
                self.display.fill((0,0,0))
                pygame.draw.rect(self.display, (255, 0, 0),pygame.Rect(self.apple[0] * 16, self.apple[1] * 16, 16, 16))
                for i in range(len(self.snake)):
                    pygame.draw.rect(self.display, (0,0,255), pygame.Rect(self.snake[i][0]*16, self.snake[i][1]*16, 16, 16))

                self.screen.blit(pygame.transform.scale(self.display, (self.screen.get_width(), self.screen.get_height()),), (0,0))

                pygame.display.update()
                self.clock.tick(10)

            return reward,death



def test():
    game = Game(True)
    agent = Agent()
    dir = 1
    total = 0
    while True:
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and dir != 4:
                    dir = 2

                elif event.key == pygame.K_DOWN and dir != 2:
                    dir = 4
                elif event.key == pygame.K_LEFT and dir != 1:
                    dir = 3
                elif event.key == pygame.K_RIGHT and dir != 3:
                    dir = 1

        reward, done =game.step(dir)
        total += reward
        if done:
            dir =1

            total = 0
            game.reset()




if __name__ == '__main__':
    test()







