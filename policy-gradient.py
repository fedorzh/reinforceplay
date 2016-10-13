""" Trains a policy-gradient agent. Uses OpenAI Gym. """
import numpy as np
import cPickle as pickle
import gym
import sys

M = 4;
I = 1000; # number of iterations
render = False;
lrate = 1e-1; # initial learning rate, for Cartpole-V0 PG only works with adaptive learning rate

theta = np.random.randn(M) * np.sqrt(2.0/M); # initialize randomly with variance recommended in http://cs231n.github.io/neural-networks-2/#summary

def logistic(x):
    return 1.0 / (1.0 + np.exp(-x));
    
def get_randomized_action(x):
    prob = logistic(np.dot(x,theta));
    action = 1 if np.random.rand() < prob else 0;
    return action, prob
    
def get_gradient(x, p, action):
    gradf = action - p; # gradient of the logistic relative to f
    return x*gradf;
    
env = gym.make("CartPole-v0");
env.monitor.start('cartpole-experiment-pg-1')
inum = 0; # iteration number
running_reward = None;
for inum in range(I):
    observation = env.reset();
    done = False;
    reward_sum = 0;
    grads = [];
    rs = [];
    while not done: # play the game with the current policy
        if render:
            env.render();
        action, prob = get_randomized_action(observation)
        grads.append(get_gradient(observation, prob, action));
        
        observation, reward, done, info = env.step(action);
        rs.append(reward);
        
    grads = np.vstack(grads);
    rs = np.vstack(rs);
    rewards = np.cumsum(rs[::-1])[::-1];
    #rewards = np.zeros_like(rewards) + sum(rs);   # this worked worse than the cumulative sum above
    gradient_with_rewards = np.dot(rewards,grads); 
    
    adaptive_lrate = lrate / (1 + inum)
    
    theta += adaptive_lrate*gradient_with_rewards;
    reward_sum = rewards[0];
    
    first_reward = reward_sum if running_reward is None else first_reward
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print 'After iteration: %d, reward sum is: %f, running reward is: %f, relative rr is: %f' % (inum, reward_sum, running_reward, running_reward/first_reward);
    
    if inum % 50 == 0:
        print '\tgradient_with_rewards is: ', gradient_with_rewards;
        print '\tTheta is: ', theta;
    
env.monitor.close()