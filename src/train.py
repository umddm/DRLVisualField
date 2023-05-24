import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.optim as optim
import torch.nn.functional as F
import argparse
import sys
import random
import glob
from matplotlib import pyplot as plt
from envs import TrainENV
from collections import namedtuple
from itertools import count
import math
import random
import numpy as np 
import time
from memory import ReplayMemory
from models import *
from multiprocessing import Pool

BATCH_SIZE = 100
GAMMA = 1.
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 1000
TARGET_UPDATE = 1000
lr = 1e-4
INITIAL_MEMORY = 1000
MEMORY_SIZE = 100 * INITIAL_MEMORY
db_low, db_high = -5, 39
state_dim = db_high - db_low + 1 # -5-39 dB  
action_dim = 41 # 0~39 dB, stop
Transition = namedtuple('Transion', ('state', 'action', 'next_state', 'reward'))

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END)* \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state.to(device)).max(1)[1].view(1,1)
    else:
        return torch.tensor([[random.randrange(action_dim)]], device=device, dtype=torch.long)
    
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    
    actions = tuple((map(lambda a: torch.tensor([[a]], device=device), batch.action))) 
    rewards = tuple((map(lambda r: torch.tensor([r], device=device), batch.reward))) 

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.bool)
    
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None]).to(device)
    
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)
    
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0] #.detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    expected_state_action_values = expected_state_action_values.unsqueeze(1)
    
    loss = F.smooth_l1_loss(state_action_values.float(), expected_state_action_values.float())
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def get_state(obs):
    state = np.array(obs)
    state = torch.from_numpy(state)
    return state.unsqueeze(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--c', default=1.0, type=float, help='Time Cost Factor Alpha')
    parser.add_argument('--l', default=10, type=int, help='ID of Testing Location')
    args = parser.parse_args()
    
    # hyperparameters
    c = args.c
    l = args.l
    
    # creating environment
    train_env = TrainENV(c=c)
    
    # create networks
    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # setup optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    
    # initialize replay memory
    memory = ReplayMemory(MEMORY_SIZE)
    
    # Save Directory
    save_model_dir = '../weights/{}/{}'.format(l, c)
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    
    # training loop
    steps_done = 0
    n_episodes = 10001
    for episode in range(1, n_episodes):
        if episode > 10000:
            lr = 1e-5
        obs = train_env.reset(l)
        state = get_state(obs)
        total_reward = 0.0
        for t in count():
            action = select_action(state)
            obs, reward, done, info = train_env.step(action.item())
            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            reward = torch.tensor([reward], device=device)
            memory.push(state, action.to('cpu'), next_state, reward.to('cpu'))
            state = next_state

            if steps_done > INITIAL_MEMORY:
                optimize_model()
                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())
            if done:
                break
                
    #torch.save(policy_net.state_dict(), os.path.join(save_model_dir, 'model.pth'))