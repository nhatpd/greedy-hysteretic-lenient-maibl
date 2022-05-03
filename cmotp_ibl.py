
from environment import Environment
from stats import mkRunDir
from config import Config
from copy import deepcopy
from itertools import count
# from agent import Agent
import csv
import json

import random as random

import os

from MAIBL import IBLAgent_td

import numpy as np
import math
from datetime import datetime
import time

import json

import matplotlib.pyplot as plt


import argparse
# import sys
# sys.argv=['']
# del sys
flags = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="lightIBL")

flags.add_argument('--environment',type=str,default='CMOTP_V3',help='Environment.')
flags.add_argument('--method',type=str,default='IBL',help='Type of Agent')
# flags.add_argument('--madrl',type=str,default='hysteretic',help='MA-IBL extension. Options: None, leniency, hysteretic')
# flags.add_argument('--madrl',type=str,default='leniency',help='MA-IBL extension. Options: None, leniency, hysteretic')
flags.add_argument('--mamethod',type=str,default='greedy',help='MA-IBL extension. Options: None, leniency, hysteretic')
# flags.add_argument('--madrl',type=str,default='original',help='MA-IBL extension. Options: None, leniency, hysteretic')
flags.add_argument('--agents',type=int,default=2,help='Number of agents.')
flags.add_argument('--episodes',type=int,default=1000,help='Number of episodes.')
flags.add_argument('--steps',type=int,default=5000,help='Number of steps.')
flags.add_argument('--default_utility',type=float,default=1,help='Number of steps.') #test for V8
FLAGS = flags.parse_args()
#V6 default = 4.4
#V8 default = 0.1 

for runid in range(1,2):
    # Environment is instantiated
    # The dimension of the obsrvations and the number 
    # of descrete actions can be accessed as follows:
    # 
    # transitions = {}
    # ttime = 0
    env = Environment(FLAGS) 

    # Example:
    config = Config(env.dim, env.out, mamethod=FLAGS.mamethod)

    # random.seed(99*runid)
    # Run dir and stats csv file are created
    statscsv, folder = mkRunDir(env, FLAGS, runid)
    # statscsv, folder = mkRunDir(env, FLAGS)

    # Agents are instantiated
    agents = []
    for i in range(FLAGS.agents): 
        agent_config = deepcopy(config)
        if FLAGS.method == "IBL":
            agents.append(IBLAgent_td(agent_config))
            # agents.append(AgentIBL(agent_config,default_utility = FLAGS.default_utility))

        f = open(folder + 'agent' + str(i) + '_config.txt','w')
        f.write(str(vars(agent_config)))
        f.close()

    sagent =[240,200]

    ##################

    # Start training run
    for i in range(FLAGS.episodes):
        
        # Run episode
        observations = env.reset() # Get first observations
        # transitions.append([np.copy(env.env.agents_x),np.copy(env.env.agents_y),np.copy([env.env.goods_x,env.env.goods_y]),False])
        # transitions[str(ttime)] = {'x1':env.env.agents_x[0], 'x2':env.env.agents_x[1],'y1':env.env.agents_y[0],'y2':env.env.agents_y[1],'x':env.env.goods_x,'y':env.env.goods_y,'t':False}
        # ttime = ttime +1
        for j in range(FLAGS.steps):

            #######################################
            # Renders environment if flag is true
            # if FLAGS.render: env.render() 

            # Load action for each agent based on o^i_t
            actions = [] 
            for agent, observation, aid in zip(agents, observations, count()):
                if FLAGS.method == "IBL":
                    actions.append(agent.move(env.env.agents_y[aid],env.env.agents_x[aid],env.env.holding_goods[aid])) 
  
            
            holding_goods = deepcopy(env.env.holding_goods)
            # Execute actions and get feedback lists:
            agents_x = deepcopy(env.env.agents_x)
            agents_y = deepcopy(env.env.agents_y)

            observations, rewards, t = env.step(actions)
            # print(observations)
            # Check if last step has been reached

            # transitions[str(ttime)] = {'x1':env.env.agents_x[0], 'x2':env.env.agents_x[1],'y1':env.env.agents_y[0],'y2':env.env.agents_y[1],'x':env.env.goods_x,'y':env.env.goods_y,'t':t}
            # ttime = ttime +1
            # transitions.append([np.copy(env.env.agents_x),np.copy(env.env.agents_y),np.copy([env.env.goods_x,env.env.goods_y]),t])
            for agent, o, r, action, x, y, aid in zip(agents, observations, rewards, actions, agents_x, agents_y, count()):
                dx, dy = env.env.getDelta(action)
                if y + dy > env.env.c.GH-1 or x + dx > env.env.c.GW-1 or o[y + dy, x + dx]== env.env.c.OBSTACLE:
                    r = -0.05
                if action == 4:
                    r = -0.01

                agent.feedback(r, t, env.env.agents_y[aid],env.env.agents_x[aid],env.env.holding_goods[aid]) 

            if j == FLAGS.steps-1:
                t = True

            # if config.madrl=="leniency":   
            #     if t:
            #         for agent in agents:
            #             print(agent.leniency)
            if t: break # If t then terminal state has been reached
        
        # Add row to stats: 
        # with open(statscsv, 'a') as csvfile:
        #     writer = csv.DictWriter(csvfile, fieldnames=env.fieldnames)
        #     writer.writerow(env.stats())
        print(env.stats())

