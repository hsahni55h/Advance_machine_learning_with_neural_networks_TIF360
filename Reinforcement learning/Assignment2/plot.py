import matplotlib.pyplot as plt
import pickle
import numpy as np

param_set = '1c'

param_map = {
    '1a': 'trial/1a/Rewards_1000.p',
    '1b': 'trial/1b/Rewards_10000.p',
    '1c': 'trial/1c/Rewards_200000.p',
    # '2a': 'reward_tots_10000.p'
}

file_name = param_map.get(param_set)

with open(file_name, "rb") as rewards_file:
    rewards_array = pickle.load(rewards_file)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

rewards_100_array = moving_average(rewards_array, 100)

plt.title("Rewards v/s episodes")
plt.plot(rewards_array, label="Instantaneous Rewards")
plt.plot(rewards_100_array, label="100 episodes moving average Rewards")
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.legend()
plt.show()
