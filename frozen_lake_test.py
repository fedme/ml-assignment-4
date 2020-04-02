import numpy as np
from gym.envs.toy_text.frozen_lake import generate_random_map, FrozenLakeEnv
import algorithms
import qlearning


# TODO: plot different map sizes vs max_val (or other stats)
# TODO: measure and plot learning times (vs map size?)
# TODO: plot maps
# TODO: plot policies
# TODO: plot max values vs n_iteration for every algorithm
# TODO: plot algorithms vs random choice


def score_frozen_lake(env, policy, episodes=1000):
    # From https://github.com/realdiganta/solving_openai
    misses = 0
    steps_list = []
    for episode in range(episodes):
        observation = env.reset()
        steps = 0
        while True:
            action = policy[observation]
            observation, reward, done, _ = env.step(action)
            steps += 1
            if done and reward == 1:
                steps_list.append(steps)
                break
            elif done and reward == 0:
                misses += 1
                break

    mean_number_of_steps = 0 if steps_list == [] else np.mean(steps_list)
    print('----------------------------------------------')
    print('You took an average of {:.0f} steps to get the frisbee'.format(mean_number_of_steps))
    print('And you fell in the hole {:.2f} % of the times'.format((misses / episodes) * 100))
    print('----------------------------------------------')


def try_value_iteration(lake_map):
    print('Trying frozen lake with Value Iteration')
    env = FrozenLakeEnv(desc=lake_map)
    optimal_policy, optimal_value_function = algorithms.value_iteration(env, theta=0.0000001, discount_factor=0.999)
    optimal_policy_flat = np.where(optimal_policy == 1)[1]
    # optimal_policy_reshaped = optimal_policy_flat.reshape(4, 4)

    score_frozen_lake(env, optimal_policy_flat)


def try_policy_iteration(lake_map):
    print('Trying frozen lake with Policy Iteration')
    env = FrozenLakeEnv(desc=lake_map)
    optimal_policy, optimal_value_function = algorithms.policy_improvement(env, discount_factor=0.9999)
    optimal_policy_flat = np.where(optimal_policy == 1)[1]
    # optimal_policy_reshaped = optimal_policy_flat.reshape(4, 4)

    score_frozen_lake(env, optimal_policy_flat)


def try_qlearning(lake_map):
    print('Trying frozen lake with Policy Iteration')
    env = FrozenLakeEnv(desc=lake_map)
    optimal_policy, optimal_q_table = qlearning.q_learning(env, 0.6, 0.95, 0.3, 10000)
    optimal_policy_flat = np.where(optimal_policy == 1)[1]
    # optimal_policy_reshaped = optimal_policy_flat.reshape(4, 4)

    score_frozen_lake(env, optimal_policy_flat)


if __name__ == '__main__':
    np.random.seed(64)
    frozen_lake_map = generate_random_map(size=8, p=0.8)
    #try_value_iteration(frozen_lake_map)
    #try_policy_iteration(frozen_lake_map)
    try_qlearning(frozen_lake_map)
