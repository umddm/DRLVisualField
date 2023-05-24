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

def load_data():    
    normal_file_test = '../data/test/normal.txt'
    normal_sens_test = []
    f = open(normal_file_test, 'r')
    for line in f.readlines():
        line = np.array(eval(line))
        normal_sens_test.append(list(line[locations]))
    f.close()
    
    abnormal_file_test = '../data/test/abnormal.txt'
    abnormal_sens_test = []
    f = open(abnormal_file_test, 'r')
    for line in f.readlines():
        line = np.array(eval(line))
        abnormal_sens_test.append(list(line[locations]))
    f.close()
    return normal_sens_test, abnormal_sens_test

def create_nn_graph():
    def find_edges_nn(points):
        edges = dict()
        for (x,y) in points:
            edges[(x,y)] = []
            for point in [(x-1, y-1), (x-1, y), (x, y-1), (x-1, y+1), (x+1, y-1), (x+1, y), (x+1, y+1), (x, y+1)]:
                if point in points:
                    edges[(x,y)].append(point)
        return edges
    points = [(7,1), (6,1), (8,0), (7,0), (6,0)]
    edges = find_edges_nn(points)
    return edges, points

def responseProb(stim, sens):
    FP, FN = 0.03, 0.03 # FP: patient does not see the stimulus, but responses accidently
    SD = min(np.exp(-0.081 * sens + 3.27), 6)
    response_prob =  1 - FN - (1 - FN - FP) * norm.cdf(stim,loc=sens,scale=SD) # cdf: cummulative density distribution
    return response_prob

def responseOrNot(stim, sens):
    response_prob =  responseProb(stim, sens) # probability between 0 and 1
    random_number = np.random.uniform(0, 1) # gaussian distributed between 0 and 1
    if random_number < response_prob and sens > 0:
        return True
    else:
        return False

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
        time = max(0.05, np.random.choice(np.arange(0,1.6,0.01), p=ps))
        return time
        
    d = stim - sens
    time = GammaFunc(d) if response else 1.5
    if time > 1.5:
        time = 1.5
        response = False
    return time, response

def get_state(obs):
    state = np.array(obs)
    state = torch.from_numpy(state)
    return state.unsqueeze(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DCorr Testing')
    parser.add_argument('--c', default=1., type=float, help='Time Cost Factor Alpha')
    args = parser.parse_args()
    
    # hyperparameters
    c = args.c
    
    # alpha to delta; these numbers are obtained by comparing DREST and ZEST
    if args.c == 0.5:
        sd_crit = 2.45
    elif args.c == 1.0:
        sd_crit = 2.82
    else:
        sd_crit = 3.27
        
    # Load data and graph
    normal_test, abnormal_test = load_data()
    edges, points = create_nn_graph()
    
    # Load models
    models = dict()
    policy_net = DQN(state_dim, action_dim)
    for i,location in enumerate(locations):
        model_path = '../weights/{}/{}/model.pth'.format(location, c)
        policy_net_copy = copy.deepcopy(policy_net)
        policy_net_copy.load_state_dict(torch.load(model_path, map_location=device))
        policy_net_copy.to(device)
        models[points[i]] = policy_net_copy
        
    def sim_exp(VF):
        pdfs, sens_graph, pdf_stds = dict(), dict(), dict()
        for i,point in enumerate(points):
            sens_graph[point] = VF[i]
            pdfs[point] = np.load('../data/train/PMF_VF/pmf%d.npy'%locations[i])
            pdf_mean = np.sum(np.arange(db_low, db_high + 1) * pdfs[point])
            pdf_stds[point] = np.sqrt(np.sum(pdfs[point] * (np.arange(db_low, db_high + 1) - pdf_mean) ** 2))
        n_presentation, total_time = 0, 0
        points_remain = list(sens_graph.keys())
        while True:
            if len(points_remain) == 0:
                break
            point = points_remain[0]
            for p in pdf_stds:
                if p in points_remain and pdf_stds[p] > pdf_stds[point]:
                    point = p
            if pdf_stds[point] > sd_crit and n_presentation <= 50:
                policy_net = models[point]
                state = get_state(pdfs[point])
                stim = int(policy_net(state).max(1)[1].view(1,1))
                if stim != action_dim - 1:
                    likelihood = np.zeros(db_length)
                    response = responseOrNot(stim, sens_graph[point])
                    time, response = responseTime(stim, sens_graph[point], response)
                    if response:
                        for i in range(db_length):
                            likelihood[i] = responseProb(stim, i + db_low)
                    else:
                        for i in range(db_length):
                            likelihood[i] = 1 - responseProb(stim, i + db_low)
                    pdf = pdfs[point] * likelihood
                    pdf = pdf / np.linalg.norm(pdf, ord=1)
                    pdfs[point] = pdf
                    pdf_mean = np.sum(np.arange(db_low, db_high + 1) * pdf)
                    pdf_stds[point] = np.sqrt(np.sum(pdf * (np.arange(db_low, db_high + 1) - pdf_mean) ** 2))
                    likelihood_connect = (likelihood - 0.5) * 0.2 + 0.5
                    for point_connect in edges[point]:
                        if point_connect in points_remain:
                            pdf = pdfs[point_connect] * likelihood_connect
                            pdf = pdf / np.linalg.norm(pdf, ord=1)
                            pdfs[point_connect] = pdf
                            pdf_mean = np.sum(np.arange(db_low, db_high + 1) * pdf)
                            pdf_stds[point_connect] = np.sqrt(np.sum(pdf * (np.arange(db_low, db_high + 1) - pdf_mean) ** 2))
                    n_presentation += 1
                    total_time += time
                    for p in pdf_stds:
                        if pdf_stds[p] <= sd_crit and p in points_remain:
                            points_remain.remove(p)
                else:
                    points_remain.remove(point)
            else:
                break
        errors, preds = [], dict()
        for point in pdfs:
            pred = max(0, np.sum(np.arange(db_low, db_high + 1) * pdfs[point]))
            preds[point] = pred
            error = np.abs(sens_graph[point] - pred)
            errors.append(error)
        return np.mean(errors), n_presentation / len(locations), total_time / len(locations)
    
    # write to logs
    log_dir = '../logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'DCorr_{}.txt'.format(c))
    f = open(log_file, 'w', buffering=1)

    with Pool(processes=64) as pool:
        result_n = pool.map(sim_exp, normal_test*10)
    result_n = np.array(result_n)
    errors_n = result_n[:,0]
    ns_n = result_n[:,1]
    ts_n = result_n[:,2]
    with Pool(processes=64) as pool:
        result_a = pool.map(sim_exp, abnormal_test*10)
    result_a = np.array(result_a)
    errors_a = result_a[:,0]
    ns_a = result_a[:,1]
    ts_a = result_a[:,2]
    
    # Normal
    f.write('MAE Normal = %.3f \n'%np.mean(errors_n))
    f.write('N Normal = %.3f \n'%np.mean(ns_n))
    f.write('T Normal = %.3f \n'%np.mean(ts_n))
    
    # Abnormal
    f.write('MAE Abnormal = %.3f \n'%np.mean(errors_a))
    f.write('N Abnormal = %.3f \n'%np.mean(ns_a))
    f.write('T Abnormal = %.3f \n'%np.mean(ts_a))
    f.close()