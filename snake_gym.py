import gym
import numpy as np

width, height = 10, 10


class Snake(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        self.action_space = gym.spaces.Discrete(3)  # Left, Straight, Right
        self.observation_space = gym.spaces.MultiDiscrete([4 for i in range(width*height)]) # nothing, snake head, snake body, bonus
        # (x, y) distance to the bonus, distance to nearest obstacle in 4 directions, direction
        self.observation_space = gym.spaces.MultiDiscrete([2*width, 2*height, width+1, height+1, width+1, height+1, 4])
        self.reward_range = (0, width*height - 3)

    def reset(self):
        self.snake = [(width//2 - 1, height//2), (width//2 - 2, height//2)]
        self.snake_head = (width//2, height//2)
        bonus = (np.random.randint(width), np.random.randint(height))
        while bonus == self.snake_head or bonus in self.snake:
            bonus = (np.random.randint(width), np.random.randint(height))
        self.bonus = bonus
        self.direction = 2  # Down
        return self.get_obs()

    def get_obs(self):
        dist_x = self.bonus[0]-self.snake_head[0]+width
        dist_y = self.bonus[1]-self.snake_head[1]+height
        for i in range(width+1):
            if (self.snake_head[0]-i)==-1 or (self.snake_head[0]-i, self.snake_head[1]) in self.snake:
                dist_left = i
                break
        for i in range(height+1):
            if (self.snake_head[1]-i)==-1 or (self.snake_head[0], self.snake_head[1]-i) in self.snake:
                dist_up = i
                break
        for i in range(width+1):
            if (self.snake_head[0]+i)==width or (self.snake_head[0]+i, self.snake_head[1]) in self.snake:
                dist_right = i
                break
        for i in range(height+1):
            if (self.snake_head[1]+i)==height or (self.snake_head[0], self.snake_head[1]+i) in self.snake:
                dist_down = i
                break
        return dist_x, dist_y, dist_left, dist_up, dist_right, dist_down, self.direction


    def get_obs_raw(self):
        curr_obs = [0 for i in range(width*height)]
        curr_obs[self.snake_head[0]*width + self.snake_head[1]] = 1
        for snake_block in self.snake:
            curr_obs[snake_block[0] * width + snake_block[1]] = 2
        curr_obs[self.bonus[0]*width + self.bonus[1]] = 3
        return curr_obs

    def step(self, action):
        reward = 0  # (15-np.sqrt((self.snake_head[0]-self.bonus[0])**2+(self.snake_head[1]-self.bonus[1])**2))/500
        self.direction += action+3
        self.direction = self.direction % 4
        if self.direction == 0:
            new_sh = (self.snake_head[0] - 1, self.snake_head[1])
        if self.direction == 1:
            new_sh = (self.snake_head[0], self.snake_head[1] - 1)
        if self.direction == 2:
            new_sh = (self.snake_head[0] + 1, self.snake_head[1])
        if self.direction == 3:
            new_sh = (self.snake_head[0], self.snake_head[1] + 1)
        if new_sh[0] < 0 or new_sh[0] >= width or new_sh[1] < 0 or new_sh[1] >= height:
            return self.get_obs(), -0.1, True, {}


        if new_sh == self.bonus:
            reward += 1
            snake_copy = self.snake[:]
            self.snake = [self.snake_head] + snake_copy
            self.snake_head = new_sh
            while self.bonus == self.snake_head or self.bonus in self.snake:
                self.bonus = (np.random.randint(width), np.random.randint(height))
            return self.get_obs(), reward, False, {}

        snake_copy = self.snake[:-1]
        self.snake = [self.snake_head] + snake_copy

        if new_sh in self.snake:
            return self.get_obs(), -0.1, True, {}

        self.snake_head = new_sh
        return self.get_obs(), reward, False, {}

    def render(self, mode='human'):
        print('-'*50)
        obs = self.get_obs_raw()
        obs = [" " if x==0 else "o" if x==1 else "#" if x==2 else "*" for x in obs ]
        obs = ''.join(obs)
        for i in range(height):
            psdline = obs[width*i:width*i+height]
            print(psdline)

# env = Snake()
#
# for i_ep in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("ep f af {}".format(t))
#             break
# env.close()



