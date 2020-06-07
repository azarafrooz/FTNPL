# No-regret dynamics/trajectories for matching pennis game:.
# matching pennies game
# (x 1-x)((1 -1)(-1 1))(y 1-y)^T
# l1= x^T*A*Y l2=-x^T*A*Y


import numpy as np
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Normal

from collections import deque

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

from tqdm import tqdm

manualSeed = 1000
print("Random Seed: ", manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)

A = torch.from_numpy(np.array([[1, -1], [-1, 1]])).float()  # gets uses for utility formation of matching game

code_dim = 2
m_lr = 0.1  # mediator learning rate
x_lr = 0.01  # lr rate for player x
y_lr = 0.01

K = 10 # size of the queue of strategy
NUM_STEPS = 5000
Sample_fr = 1 # downsampling for visualization

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Mediator(nn.Module):
    '''
    Mediator/correlator intervenes with the learning dynamics
    '''
    def __init__(self, input_dim, action_dim):
        super(Mediator, self).__init__()
        self.embed_mean = nn.Linear(input_dim, action_dim)
        self.embed_log_std = nn.Linear(input_dim, action_dim)

        self.apply(weights_init_)

        self.action_scale = torch.tensor(1)
        self.action_bias = torch.tensor(0.0)

    def forward(self, x_player_action, y_player_action):
        obs = torch.cat((x_player_action, y_player_action), dim=1)
        mean, log_std = self.embed_mean(obs), self.embed_log_std(obs)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        std = torch.exp(log_std)
        return mean, std

    def act(self, x_player_action, y_player_action):
        '''
        pathwise derivative estimator for taking actions.
        :param x_player_action:
        :param y_player_action:
        :return:
        '''
        mean, std = self.forward(x_player_action, y_player_action)
        normal = Normal(mean, std)
        x = normal.rsample()
        y = torch.tanh(x)
        action = y*self.action_scale + self.action_bias
        log_prob = normal.log_prob(action)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y.pow(2)) + EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class Player(nn.Module):
    '''
    Abstraction for Players
    '''
    def __init__(self, code_dim):
        super(Player, self).__init__()
        self.linear = nn.Linear(code_dim, 2)

    def forward(self, code):
        return F.softmax(self.linear(code), dim=1)-0.5


pbar = tqdm(total=NUM_STEPS)

x_queue = deque(maxlen=K)  # queue of past strategies for player x
y_queue = deque(maxlen=K)  # queue of past strategies for player y


mediator = Mediator(4, code_dim)  # gets x,y with dim (1,2), y with dim (1,2) and returns code_dim

x_player = Player(code_dim)
y_player = Player(code_dim)

x_queue.append(copy.deepcopy(x_player.state_dict()))
y_queue.append(copy.deepcopy(y_player.state_dict()))

m_optimizer = optim.SGD(mediator.parameters(), lr=m_lr)
x_optimizer = optim.SGD(x_player.parameters(), lr=x_lr)
y_optimizer = optim.SGD(y_player.parameters(), lr=y_lr)

strategies = []

code = torch.zeros(1, code_dim)
x, y = x_player(code), y_player(code)

for t in range(NUM_STEPS):
    # mediator proposes a code
    code, log_intervense_prob, mean = mediator.act(x.detach(), y.detach())
    code = mean.detach()
   
    # player x's turn
    x_optimizer.zero_grad()
    x_loss = torch.zeros(1, 1)
    y = y_player(code)
    x_gains = []

    for param in x_queue:
        x_player.load_state_dict(param)
        x = x_player(code)
        loss = torch.mm(torch.mm(x, A), y.T)
        x_gains.append(loss.item())
        x_loss += loss

    x_loss /= len(x_queue)
    x_loss.backward()
    x_optimizer.step()
    x_queue.append(copy.deepcopy(x_player.state_dict()))

    # player y's turn
    y_optimizer.zero_grad()
    y_loss = torch.zeros(1, 1)
    x = x_player(code)
    y_gains = []

    for param in y_queue:
        y_player.load_state_dict(param)
        y = y_player(code)
        loss = -torch.mm(torch.mm(x, A), y.T)
        y_gains.append(loss.item())
        y_loss += loss

    y_loss /= len(y_queue)
    y_loss.backward()
    y_optimizer.step()
    y_queue.append(copy.deepcopy(y_player.state_dict()))

    # mediator updates
    m_optimizer.zero_grad()
    # computing reward for mediator
    reward = torch.tensor(0.0)
    for i in range(len(x_gains)):
        for j in range(i + 1, len(x_gains)):
            reward -= np.max((x_gains[i] - x_gains[j]), 0)
            reward -= np.max((y_gains[i] - y_gains[j]),0)

    reward = reward/(len(x_gains) * len(x_gains) / 4)

    mediator_loss = -log_intervense_prob * reward
    mediator_loss.backward()
    m_optimizer.step()
    # optimization done

    strategies.append(
        [x[0][0].item(), y[0][0].item()])  # building trajectory for vis. Only the first strategy is plotted.

    # print(torch.mm(torch.mm(x, A), y.T))
    pbar.update(1)

strategies = np.array(strategies)[0::Sample_fr]

x = strategies[:-1, 0]
y = strategies[:-1, 1]
u = strategies[1:, 0] - strategies[:-1, 0]
v = strategies[1:, 1] - strategies[:-1, 1]
n = 0
color_array = np.sqrt(((u-n))**2 + ((v-n))**2)
norm = colors.Normalize(color_array.min(), color_array.max())
color_array = cm.jet(norm(color_array))

plt.quiver(x,y,u,v, linewidth=0.0001, pivot='middle', color=color_array)

plt.ioff()
plt.plot()
plt.savefig("ftnpl")
print("saved the figure under the ftnpl.png")












