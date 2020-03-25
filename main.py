import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd
import snake_game
import matplotlib.pyplot as plt
import argparse

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Multiple object tracker')
    parser.add_argument('--rl_algorithm', type=str)
    parser.add_argument('--explore_method', type=str)
    parser.add_argument('--n_episode', type=int)
    parser.add_argument('--training', type=bool)
    parser.add_argument('--n_env', type=int)
    args = parser.parse_args()
    return args

import snake_gym_env_1
import snake_gym_env_2

args = parse_args()
# Meta parameters for the RL agent
alpha = 0.1
tau = init_tau = 1
tau_inc = 0.01
gamma = 0.99
epsilon = 0.5
epsilon_decay = 0.99997
verbose = True

# Define types of algorithms
SARSA = "SARSA"
Q_LEARNING = "Q_learning"
EPSILON_GREEDY = "epsilon_greedy"
SOFTMAX = "softmax"
GREEDY = "greedy"
VDBE_EPSILON = "vdbe_epsilon"

# Choose methods for learning and exploration
rl_algorithm = args.rl_algorithm
explore_method = args.explore_method
n_episode = args.n_episode
train = args.training
print(rl_algorithm)

#Parameters for VDBE
sigma = 10
delta=1/3
n_env = args.n_env


# Draw a softmax sample
def softmax(q):
    assert tau >= 0.0
    q_tilde = q - np.max(q)
    factors = np.exp(tau * q_tilde)
    return factors / np.sum(factors)

# Act with softmax
def act_with_softmax(s, q):
    prob_a = softmax(q[s])
    cumsum_a = np.cumsum(prob_a)
    return np.where(np.random.rand() < cumsum_a)[0][0]

# Act with epsilon greedy
def act_with_epsilon_greedy(s, q, e):
    a = np.argmax(q[s])
    if np.random.rand() < e:
        a = np.random.randint(q.shape[-1])
        return a, 1
    return a, 0

def act_with_epsilon_softmax(s,q,e):
    a = np.argmax(q[s])
    if np.random.rand() < e:
        a = act_with_softmax(s,q)
        return a, 1
    return a, 0

#Update value of the epsilon table for VBED
def update_epsilon_vdbe(old_q,new_q,old_epsilon):
    if old_q==0 and new_q==0:
        return old_epsilon
    delta_q = abs(old_q-new_q)
    expo = np.exp(-delta_q/sigma)
    f = (1-expo)/(1+expo)
    return delta * f + (1-delta)*old_epsilon

# Compute SARSA update
def sarsa_update(q,s,a,r,s_prime,a_prime):
    prime = s_prime + tuple([a_prime])
    base = s + tuple([a])
    td = r + gamma * q[prime] - q[base]
    return q[base] + alpha * td

# Compute Q-Learning update
def q_learning_update(q,s,a,r,s_prime):
    base = s + tuple([a])
    td = r + gamma * np.max(q[s_prime]) - q[base]
    return q[base] + alpha * td


def main():

    global epsilon
    global tau

    np.random.RandomState(42)

    if n_env==1:
        env = snake_gym_env_1.Snake()
    if n_env==2:
        env = snake_gym_env_2.Snake()

    # Recover State-Action space size
    n_a = env.action_space.n
    list_s = env.observation_space.nvec
    dim = list_s[:].tolist()
    dim.append(n_a)

    # Experimental setup
    print("n_episode ", n_episode)
    max_horizon = 500
    eval_steps = 10

    if train:

        # Init Q-table

        q_table = np.zeros(dim)
        env.reset()

        # Init epsilon table for vdbe
        epsilon_table = np.ones(list_s)

        X=[]
        Y=[]
        Y_mean=[]
        P=[]

        #init parameters graph
        mean_return = 0.0
        interval_graph = 50

        #calculate rate of exploration
        n_exploration=0
        n_tot=0
        # Train for n_episode
        for i_episode in range(n_episode):
            
            # Reset a cumulative reward for this episode
            total_return = 0.0
            

            # Start a new episode and sample the initial state
            s = env.reset()

            # Select the first action in this episode
            if explore_method == SOFTMAX:
                a = act_with_softmax(s, q_table)
            elif explore_method == EPSILON_GREEDY:
                a, exploration = act_with_epsilon_greedy(s, q_table,epsilon)
                n_exploration+=exploration
                n_tot+=1
            elif explore_method == VDBE_EPSILON:
                a, exploration = act_with_epsilon_softmax(s,q_table,epsilon_table[s])
                n_exploration+=exploration
                n_tot+=1
            else:
                raise ValueError("Wrong Explore Method:".format(explore_method))
            

            for i_step in range(max_horizon):
                
                # Act
                s_prime, r, done, info = env.step(a)
                #count the number of eaten apple
                if r>1:
                    total_return += 1

                # Select an action
                if explore_method == SOFTMAX:
                    a_prime = act_with_softmax(s_prime, q_table)
                elif explore_method == EPSILON_GREEDY:
                    a_prime, exploration = act_with_epsilon_greedy(s_prime, q_table,epsilon)
                    n_exploration+=exploration
                    n_tot+=1
                elif explore_method == VDBE_EPSILON:
                    a_prime, exploration = act_with_epsilon_softmax(s_prime, q_table,epsilon_table[s_prime])
                    n_exploration+=exploration
                    n_tot+=1
                else:
                    raise ValueError("Wrong Explore Method:".format(explore_method))

                #Store old q value
                base = s + tuple([a])
                old_q = q_table[base]
                # Update a Q value table
                if rl_algorithm == SARSA:
                    base = s + tuple([a])
                    q_table[base] = sarsa_update(q_table,s,a,r,s_prime,a_prime)
                elif rl_algorithm == Q_LEARNING:
                    base = s + tuple([a])
                    q_table[base] = q_learning_update(q_table,s,a,r,s_prime)
                else:
                    raise ValueError("Wrong RL algorithm:".format(rl_algorithm))

                #Update epsilon_table
                if explore_method == VDBE_EPSILON:
                    new_q = q_table[base]
                    epsilon_table[s] = update_epsilon_vdbe(old_q,new_q,epsilon_table[s])

                # Transition to new state
                s = s_prime
                a = a_prime
                if done:
                    break
                    
            if i_episode % 1000== 0:
                if verbose:
                    if explore_method == VDBE_EPSILON:
                        print("Episode: {0}\t Num_Steps: {1:>4}\tMean_Return:{2:>5.2f}\tTotal_Return: {3:>5.2f}\tMean_epsilon: {4:.3f}".format(i_episode, i_step,mean_return, total_return, np.mean(epsilon_table)))
                    else :    
                        print("Episode: {0}\t Num_Steps: {1:>4}\tMean_Return:{2:>5.2f}\tTotal_Return: {3:>5.2f}\tEpsilon: {4:.3f}".format(i_episode, i_step,mean_return, total_return, epsilon))
                    #print "Episode: {0}\t Num_Steps: {1:>4}\tTotal_Return: {2:>5.2f}\tTermR: {3}\ttau: {4:.3f}".format(i_episode, i_step, total_return, r, tau)


            # Schedule for epsilon
            epsilon = epsilon * epsilon_decay
            # Schedule for tau
            tau = init_tau + i_episode * tau_inc

            #draw graph
            mean_return+=total_return/interval_graph
            if i_episode%interval_graph==0:
                X.append(i_episode)
                Y.append(total_return)
                Y_mean.append(mean_return)
                mean_return=0
                if explore_method==EPSILON_GREEDY or explore_method == VDBE_EPSILON:
                    P.append(n_exploration/n_tot)
                    n_exploration=0
                    n_tot=0


        q_flat = q_table.flatten()
        np.save(f'RESULTS/q_saved_{rl_algorithm}_{explore_method}_{n_episode}_episodes_env_{n_env}.npy', q_flat)

        #Graph of results
        plt.xlabel('Episodes')
        plt.ylabel('Score')
        plt.title(f'Results with {rl_algorithm} algorithm and {explore_method} in env {n_env}')
        plt.plot(X,Y,'ro',label='Total_return')
        plt.plot(X,Y_mean,'b+',label='Mean return')
        plt.legend()
        plt.savefig(f'RESULTS/graph_{rl_algorithm}_{explore_method}_{n_episode}_episodes_env_{n_env}.png')
        plt.show()
        if explore_method==EPSILON_GREEDY or explore_method == VDBE_EPSILON:
            plt.xlabel('Episodes')
            plt.ylabel('Exploration rate')
            plt.title(f'Exploration rate with {rl_algorithm} algorithm and {explore_method} in env {n_env}')
            plt.plot(X,P)
            plt.savefig(f'RESULTS/courbe_epsilon_{rl_algorithm}_{explore_method}_{n_episode}_episodes_env_{n_env}.png')
            plt.show()


    q_table = np.load(f'RESULTS/q_saved_{rl_algorithm}_{explore_method}_{n_episode}_episodes_env_{n_env}.npy').reshape(dim)

    actions = []
    rew = []
    bonus = []
    s = env.reset()
    for step in range(max_horizon):
        env.render()
        act = np.argmax(q_table[s])
        s, r, done, info = env.step(act)
        actions.append(act)
        bonus.append(env.bonus)
        rew.append(r)
        if done:
            break

    snake_game.main(actions, 10, 10, bonus)
    exit(0)

if __name__ == "__main__":

    main()
