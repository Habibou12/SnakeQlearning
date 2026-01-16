from collections import deque
import random
from sympy.physics.units import tonne
import torch
from Model import *

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Qnetwork()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=LR)
        self.criterion = torch.nn.MSELoss()

    def get_state(self, game):
        head = game.snake[0]
        apple = game.apple
        dir = game.dir
        lefth = [head[0] - 1, head[1]]
        righth = [head[0] + 1, head[1]]
        downh = [head[0] , head[1] + 1]
        uph = [head[0] , head[1] - 1]
        frontcol = False
        if dir == 1 and game.is_collision(righth) or dir == 2 and game.is_collision(uph) or dir == 3 and game.is_collision(lefth) or dir == 4 and game.is_collision(downh):
            frontcol = True
        leftcol = False
        if dir == 2 and game.is_collision(lefth) or dir == 1 and game.is_collision(uph) or dir == 3 and game.is_collision(downh) or dir == 4 and game.is_collision(righth):
            leftcol = True
        rightcol = False
        if dir == 2 and game.is_collision(righth) or dir == 1 and game.is_collision(downh) or dir == 3 and game.is_collision(uph) or dir == 4 and game.is_collision(lefth):
            rightcol = True

        dir_r = dir == 1
        dir_l = dir == 3
        dir_u = dir == 2
        dir_d = dir == 4

        apple_r = apple[0] > head[0]
        apple_l = apple[0] < head[0]
        apple_d = apple[1] > head[1]
        apple_u = apple[1] < head[1]

        return frontcol, leftcol, rightcol, dir_r, dir_l, dir_u, dir_d, apple_r, apple_l, apple_d,apple_u


    def act(self, game, eps):
        state = self.get_state(game)
        state = torch.tensor(state, dtype=torch.float)
        self.model.eval()
        with torch.no_grad():
            action = torch.argmax(self.model(state)).item()
        self.model.train()
        if np.random.random() < eps:
            return np.random.randint(0,3)
        return action

    def remember(self, state, action, reward, next_state,done):
        self.memory.append((state, action, reward, next_state,done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
            state, action, reward, next_state, done = zip(*mini_sample)
            self.train_step(state, action, reward, next_state, done)
    def train_short_memory(self, state, action, reward, next_state, done):
        self.train_step(state, action, reward, next_state, done)



    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)
        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][action[idx]] = Q_new

            # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
            # pred.clone()
            # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()








