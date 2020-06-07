# No-regret dynamics/trajectories for matching pennies game:
# matching pennis game
# (x 1-x)((1 -1)(-1 1))(y 1-y)^T
# l1= x^T*A*Y l2=-x^T*A*Y 

import numpy as np
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.distributions.exponential import Exponential

from utils import queue_update

from collections import deque

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

from tqdm import tqdm

A = torch.from_numpy(np.array([[1, -1], [-1, 1]])).float()  # gets uses for utility formation of matching game

NUM_STEPS = 10000
x_lr = 0.01
y_lr = 0.005
K = 10  # size of the queue of strategy
m = 1  # m & inc control the pruning
inc = 10  # m & inc control the pruning

# randomness used for perturbations
x_rate = 5 # rate of the exponential distribution perturbation
y_rate = 1

Sample_fr = 1 # downsampling for visualization
pbar = tqdm(total=NUM_STEPS)



class Player(nn.Module):
    '''
    Abstraction for Players
    '''
    def __init__(self):
        super(Player, self).__init__()
        self.linear = nn.Parameter(torch.rand(1, 2))

    def forward(self):
        return F.softmax(self.linear, dim=1)-0.5


x_queue = deque(maxlen=K)  # queue of past strategies for player x
y_queue = deque(maxlen=K)  # queue of past strategies for player y

x_player = Player()
y_player = Player()


x_queue, m = queue_update(queue=x_queue, m=m, K=K, t=0, ft=copy.deepcopy(x_player.state_dict()), inc=inc)
y_queue, _ = queue_update(queue=y_queue, m=m, K=K, t=0, ft=copy.deepcopy(y_player.state_dict()), inc=inc)

x_optimizer = optim.SGD(x_player.parameters(), lr=x_lr)
y_optimizer = optim.SGD(y_player.parameters(), lr=y_lr)

strategies = []
params = torch.Tensor()
for param in x_player.parameters(): # the same number of params for y_player
    params = torch.cat((params, param.view(-1)))

for t in range(NUM_STEPS):
    # player x's turn
    x_optimizer.zero_grad()
    x_loss = torch.zeros(1, 1)
    y = y_player()
    for param in x_queue:
        x_player.load_state_dict(param)
        x = x_player()
        x_loss += torch.mm(torch.mm(x, A), y.T)
    x_loss /= len(x_queue)

    # perturbation
    randomness = Exponential(x_rate*torch.ones(len(params))).sample()
    params = torch.Tensor()
    for param in x_player.parameters():
        params = torch.cat((params, param.view(-1)))
    x_loss -= torch.dot(params, randomness)

    x_loss.backward()
    x_optimizer.step()
    x_queue, m = queue_update(queue=x_queue, m=m, K=K, t=t+1, ft=copy.deepcopy(x_player.state_dict()), inc=inc)

    # player y's turn
    y_optimizer.zero_grad()
    x = x_player()
    y_loss = torch.zeros(1, 1)
    for param in y_queue:
        y_player.load_state_dict(param)
        y = y_player()
        y_loss += torch.relu(-torch.mm(torch.mm(x, A), y.T))
    y_loss /= len(y_queue)

    # perturbation
    randomness = Exponential(y_rate*torch.ones(len(params))).sample()
    params = torch.Tensor()
    for param in y_player.parameters():
        params = torch.cat((params, param.view(-1)))
    y_loss -= torch.dot(params, randomness)

    y_loss.backward()
    y_optimizer.step()
    y_queue, _ = queue_update(queue=y_queue, m=m, K=K, t=t+1, ft=copy.deepcopy(y_player.state_dict()), inc=inc)
    pbar.update(1)

    strategies.append([x[0][0].item(), y[0][0].item()])  # building trajectory for vis. Only the first strategy is plotted.
   
# trajectory demonstration
strategies = np.array(strategies)[0::Sample_fr]

x = strategies[:-1, 0]
y = strategies[:-1, 1]
u = strategies[1:, 0] - strategies[:-1, 0]
v = strategies[1:, 1] - strategies[:-1, 1]
n = 0
color_array = np.sqrt(((u-n))**2 + ((v-n))**2)
norm = colors.Normalize(color_array.min(), color_array.max())
color_array = cm.jet(norm(color_array))

plt.quiver(x, y, u, v, linewidth=0.0001, pivot='middle', color=color_array)
plt.ioff()
plt.plot()
plt.savefig('ftpl_nonconvex')
print("figure saved under the ftpl_nonconvex.png")












