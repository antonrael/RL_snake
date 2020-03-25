Requirements:

numpy
gym
pygame

Instructions:

snake_game.py contains the playable version of the snake game. It can be run on its own (using python3 snake.py)

snake_gym_env_1.py and snake_gym_env2.py contain the code to adapt the snake game to the openai framework ; they are not meant to be executed separately

main.py contains the code used to train our agent. It must be executed using several options:
 --training : if set to True, the agent will be trained ; otherwise, the reward matrix is loaded. In either case, one episode is generated then displayed
 --rl_algorithm : must be set to SARSA or Q_learning, defines the algorithm that will be used
 --explore_method : must be set to epsilon_greedy or softmax, defines the exploration method
 --n_episode : an integer, defines the number of episodes to run
 --n_env : 1 or 2, defines the encoding used for the state space

Example : 
python3 main.py --rl_algorithm SARSA --explore_method softmax --n_episode 100000 --n_env 2 --training True

plot_all_graphs.py contains a code which trains directly all the agents and plot a graph to compare their training evolution. The user must precise the number of episodes --n_episode


