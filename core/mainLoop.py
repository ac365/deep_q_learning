from agent.Agent import Agent

import numpy as np
import gymnasium as gym

#init agent
actDims  = [2]
obsDims  = [4]
actSpace = np.array([0,1], dtype=np.int64)
actor = Agent(obsDims, actDims, actSpace)

#init environment
env = gym.make("CartPole-v1")
state, info = env.reset()

#act, step, memorize, learn
action = actor.actSpace[actor.act(state)]


