from game import *
from Trainer import *
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/mon_experience_1')
plt.ion()



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


def trainq(episode):
    game = Game(False)
    agent = Agent()
    eps = 0.9
    record = []
    for i in range(episode):
        game.reset()
        state = agent.get_state(game)

        episodescore = 0
        done = False
        step = 0
        while not done and step < len(game.snake) *50:
            action = agent.act(game, eps)
            dir = game.dir
            actionf = clockwise(action, dir)
            reward, done = game.step(actionf)
            step += 1
            episodescore += reward
            nextState = agent.get_state(game)
            agent.remember(state, action, reward, nextState, done)
            agent.train_short_memory(state, action, reward, nextState, done)
            state = nextState
        agent.train_long_memory()
        record.append(episodescore)
        if i %10 == 0:
            print(str(i) + ": " + str(episodescore))
            agent.save()


        writer.add_scalar('Loss/train', episodescore, i)


        if eps > 0.01:
            eps -= 0.005  # On enlève 0.05% à chaque partie
        else:
            eps = 0.01



trainq(100000)