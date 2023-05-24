import os
import argparse
import sys
from glob import glob
import numpy as np 
import time
import copy
from multiprocessing import Pool
import json
from scipy.stats import norm, gengamma
from models import *
device = torch.device("cpu")

db_low, db_high = -5, 39
state_dim = db_high - db_low + 1 # -5-39 dB  
action_dim = 41 # 0~39 dB, stop
db_length = state_dim
locations = [10,11,18,19,20]

def load_data(l):    
    normal_file_test = '../data/test/normal.txt'
    normal_sens_test = []
    f = open(normal_file_test, 'r')
    for line in f.readlines():
        line = eval(line)
        #normal_sens_test += line
        normal_sens_test.append(line[l])
    f.close()
    
    abnormal_file_test = '../data/test/abnormal.txt'
    abnormal_sens_test = []
    f = open(abnormal_file_test, 'r')
    for line in f.readlines():
        line = eval(line)
        #abnormal_sens_test += line
        abnormal_sens_test.append(line[l])
    f.close()
    return normal_sens_test, abnormal_sens_test

def responseProb(stim, sens):
    FP, FN = 0.03, 0.03
    SD = min(np.exp(-0.081 * sens + 3.27), 6)
    response_prob =  1 - FN - (1- FN -FP) * norm.cdf(stim,loc=sens,scale=SD)
    return response_prob

def responseOrNot(stim, sens):
    response_prob =  responseProb(stim, sens)
    random_number = np.random.uniform(0, 1)
    if random_number < response_prob and sens > 0:
        return 1
    else:
        return 0
            
def responseTime(stim, sens, response):
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
        for i in np.arange(0,16, 0.1):
            ps.append(gengamma.pdf(i,h,c))
        ps = ps/sum(ps)
        time = np.random.choice(np.arange(0,1.6,0.01), p=ps)
        return time
        
    d = stim - sens 
    time = GammaFunc(d) if response else 1.5
    if time > 1.5:
        time = 1.5
        response = 0
    return time, response

def get_state(obs):
    state = np.array(obs)
    state = torch.from_numpy(state)
    return state.unsqueeze(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DREST Testing')
    parser.add_argument('--c', default=1., type=float, help='Time Cost Factor')
    args = parser.parse_args()
    
    # hyperparameters
    c = args.c
    
    # creating environment
    def sim_exp(sens):
        pdf = np.load('../data/train/PMF_VF/pmf%d.npy'%location)
        pred = np.sum(np.arange(db_low, db_high + 1) * pdf)
        total_count = 0
        total_time = 0
        while total_count <= 10:
            state = get_state(pdf)
            stim = int(policy_net(state).max(1)[1].view(1,1))
            if stim != action_dim - 1:
                likelihood = np.zeros(db_length)
                response = responseOrNot(stim, sens)
                time, response = responseTime(stim, sens, response)
                if response:
                    for i in range(db_length):
                        likelihood[i] = responseProb(stim, i + db_low)
                else:
                    for i in range(db_length):
                        likelihood[i] = 1 - responseProb(stim, i + db_low)
                pdf = pdf * likelihood
                pdf = pdf / np.linalg.norm(pdf, ord=1)
                pdf_mean = np.sum(np.arange(db_low, db_high + 1) * pdf)
            else:
                pred = pdf_mean
                break
            total_count += 1
            total_time += time
        error = np.abs(sens - pred)
        return error, total_count, total_time
    
    # Create log
    log_file = '../logs/DREST_{}.txt'.format(c)
    f = open(log_file, 'w', buffering=1)
    
    # Load models
    models = dict()
    policy_net = DQN(state_dim, action_dim)
    
    # Load Validaiton and Testing data
    errors_n, errors_a, ns_n, ns_a, ts_n, ts_a = [], [], [], [], [], []
    for location in locations:
        normal_test, abnormal_test = load_data(location)
        model_path = '../weights/{}/{}/model.pth'.format(location, c)
        policy_net.load_state_dict(torch.load(model_path, map_location=device))
        policy_net.to(device)
        
        with Pool(processes=64) as pool:
            result_n = pool.map(sim_exp, normal_test*10)
        result_n = np.array(result_n)
        errors_n += list(result_n[:,0])
        ns_n += list(result_n[:,1])
        ts_n += list(result_n[:,2])
        with Pool(processes=64) as pool:
            result_a = pool.map(sim_exp, abnormal_test*10)
        result_a = np.array(result_a)
        errors_a += list(result_a[:,0])
        ns_a += list(result_a[:,1])
        ts_a += list(result_a[:,2])
    
    # Normal
    f.write('MAE Normal = %.3f \n'%np.mean(errors_n))
    f.write('N Normal = %.3f \n'%np.mean(ns_n))
    f.write('T Normal = %.3f \n'%np.mean(ts_n))
    
    f.write('MAE Abnormal = %.3f \n'%np.mean(errors_a))
    f.write('N Abnormal = %.3f \n'%np.mean(ns_a))
    f.write('T Abnormal = %.3f \n'%np.mean(ts_a))
    f.close()