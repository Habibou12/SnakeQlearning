import torch
from Trainer import *
from game import *


def clockwise(action, dir):
    clock_wise = [1, 2, 3, 4]
    idx = clock_wise.index(dir)

    if action == 0:
        new_dir = clock_wise[idx]  # no change
    elif action == 1:
        next_idx = (idx + 1) % 4
        new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
    else:  # [0, 0, 1]
        next_idx = (idx - 1) % 4
        new_dir = clock_wise[next_idx]

    return new_dir

def lauchSimulation():
    agent = Agent()
    game = Game(True)
    agent.load()
    while True:
        done = False
        step = 0
        while not done and step < len(game.snake) * 50:
            step += 1
            action = agent.act(game, 0)
            dir = game.dir
            actionf = clockwise(action, dir)
            reward, done = game.step(actionf)

lauchSimulation()
