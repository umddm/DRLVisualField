import numpy as np
import random
import scipy.stats as st
from scipy.stats import norm, gengamma

import gym
from gym import spaces
from gym.utils import seeding

class TrainENV(gym.Env):
    
    def __init__(self, db_high=39, db_low=-5, nb_actions=41, nb_states=45, max_count=10, c=1.):
        self.db_high = db_high
        self.db_low = db_low
        self.db_length = db_high - db_low + 1
        
        self.action_dim = nb_actions # [continue/stop, 0~39 dB]
        self.action_space = spaces.Box(low=np.array([0]), high=np.array([nb_actions - 1]), dtype=int) # dB: 0~39, stop
        
        self.nb_states = nb_states # [0~39 dB]
        state_high_matrix = np.ones(nb_states)
        self.observation_space = spaces.Box(low=np.array([0.] * nb_states), high=state_high_matrix, dtype=np.float32)

        self.max_count = max_count
        self.c = c

    def step(self, action):
        pred_previous = np.sum(np.arange(self.db_low, self.db_high + 1) * self.observation)
        
        if action == self.action_dim - 1: # stop signal
            stim, response = -1, -1
            reward = 0
            done = True
        else:
            stim = action
            
            pred_previous = self.pred

            observation_y = self.mle(self.observation, stim, 1)
            pred_y = np.sum(np.arange(self.db_low, self.db_high + 1) * observation_y)
            observation_n = self.mle(self.observation, stim, 0)
            pred_n = np.sum(np.arange(self.db_low, self.db_high + 1) * observation_n)
            
            response, response_prob = self.responseOrNot(stim, self.sens)
            time, response = self.responseTime(stim, self.sens, response_prob, response)
            
            self.observation = observation_y if response else observation_n
            self.pred = pred_y if response else pred_n
            
            reward = np.abs(pred_previous - self.sens) - np.abs(self.pred - self.sens) - self.c * time
            
            self.total_count += 1
            self.total_time += time
            done = self.total_count >= self.max_count
            
        return self.observation, reward, done, {"sens": self.sens, "pred": self.pred, "stim": stim, "response": response, "time": self.total_time, "count": self.total_count}
        
    def mle(self, prior, stim, response):
        likelihood = np.zeros(self.db_length)
        if response:
            for i in range(self.db_length):
                likelihood[i] = self.responseProb(stim, i + self.db_low)
        else:
            for i in range(self.db_length):
                likelihood[i] = 1 - self.responseProb(stim, i + self.db_low)
                
        posterior = prior * likelihood
        posterior = posterior / np.linalg.norm(posterior, ord=1)
        return posterior
    
    def responseProb(self, stim, sens):
        FP, FN = 0.03, 0.03
        SD = min(np.exp(-0.081 * sens + 3.27), 6)
        response_prob =  1 - FN - (1- FN -FP) * norm.cdf(stim,loc=sens,scale=SD)
        return response_prob

    def responseOrNot(self, stim, sens):
        response_prob =  self.responseProb(stim, sens)
        random_number = np.random.uniform(0, 1)
        if random_number < response_prob and sens > 0:
            return 1, response_prob
        else:
            return 0, response_prob
            
    def responseTime(self, stim, sens, response_prob, response):
        def muvarfunc(d):
            if d <= 2:
                mu = 3.86 + 1.44*np.exp(d/2.04)
                var = 0.2 + 4.0*np.exp(d/2.0)
            else:
                mu, var = muvarfunc(2)
            return mu, var
            
        def GammaFunc(d):
            mu, var = muvarfunc(d)
            h, c = mu**2/var, mu/var
            ps = []
            for i in np.arange(0,16,0.1):
                ps.append(gengamma.pdf(i,h,c))
            ps = ps/sum(ps)
            time = np.sum(np.arange(0,1.6,0.01) * ps)
            return time
            
        d = stim - sens   
        time = GammaFunc(d) * response_prob + 1.5 * (1-response_prob)
        if time > 1.5:
            time = 1.5
            response = 0
        return time, response
        
    def reset(self, location):
        self.observation = np.load('../data/train/PMF_VF/pmf%d.npy'%location)
        self.pred = np.sum(np.arange(self.db_low, self.db_high + 1) * self.observation)
        self.sens = max(0, np.random.choice(np.arange(self.db_low, self.db_high + 1), p=self.observation))
        self.total_time = 0
        self.total_count = 0
        return self.observation