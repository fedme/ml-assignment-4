import json
import numpy as np
from gym.envs.toy_text.frozen_lake import generate_random_map, FrozenLakeEnv
from timeit import default_timer as timer

from policy_iteration import policy_improvement
from qlearning import q_learning
from value_iteration import value_iteration


# TODO: plot different map sizes vs max_val (or other stats)
# TODO: measure and plot learning times (vs map size?)
# TODO: plot maps
# TODO: plot policies
# TODO: plot max values vs n_iteration for every algorithm
# TODO: plot algorithms vs random choice


MAP_SIZES = [4, 8, 12, 16, 20, 24, 28, 32]


########################################
# Environments
########################################

def find_good_maps(map_p=0.8):
    sizes = MAP_SIZES
    # sizes = [4, 8]
    seeds = range(20)
    best_maps = {}

    for size in sizes:
        smallest_lost_games_perc = float('inf')
        best_map = None
        for seed in seeds:
            print(f'Finding best maps with size {size} (seed {seed})...')
            np.random.seed(seed)
            map = generate_random_map(size=size, p=map_p)
            env = FrozenLakeEnv(desc=map)
            optimal_policy, optimal_value_function = value_iteration(env, theta=0.0000001, discount_factor=0.999)
            optimal_policy_flat = np.where(optimal_policy == 1)[1]
            mean_number_of_steps, lost_games_perc = score_frozen_lake(env, optimal_policy_flat)
            if lost_games_perc < smallest_lost_games_perc:
                smallest_lost_games_perc = lost_games_perc
                best_map = map
        best_maps[size] = {
            'lost_games_perc': smallest_lost_games_perc,
            'map': best_map
        }

    with open(f'best_maps_{map_p}.json', "wb") as f:
        f.write(json.dumps(best_maps).encode("utf-8"))
    return best_maps


def load_maps(map_p=0.8):
    with open(f'best_maps_{map_p}.json') as json_file:
        data = json.load(json_file)
        return data


def get_map(maps, map_size=8):
    return maps[f'{map_size}']['map']


def create_env(lake_map):
    return FrozenLakeEnv(desc=lake_map)


########################################
# Scoring
########################################

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
    lost_games_perc = (misses / episodes) * 100

    return mean_number_of_steps, lost_games_perc


def print_score(mean_number_of_steps, lost_games_perc):
    print('----------------------------------------------')
    print('You took an average of {:.0f} steps to get the frisbee'.format(mean_number_of_steps))
    print('And you fell in the hole {:.2f} % of the times'.format(lost_games_perc))
    print('----------------------------------------------')


########################################
# Q Learning
########################################

def run_qlearning(env, alpha=0.8, gamma=0.9999, epsilon=0.1, episodes=10000):
    optimal_policy, optimal_q_table = q_learning(env, alpha=alpha, gamma=gamma, epsilon=epsilon, episodes=episodes)
    optimal_policy_flat = np.where(optimal_policy == 1)[1]
    return optimal_policy_flat


def analyze_qlearning(map_p=0.8):
    maps = load_maps(map_p=map_p)

    map_sizes = MAP_SIZES
    n_episodes_values = [5000, 10000, 50000, 100000, 200000]
    results = []

    for map_size in map_sizes:
        map = get_map(maps, map_size=map_size)
        for n_episodes in n_episodes_values:
            print('########################################')
            print(f'Running qlearning for map_size={map_size} and n_episodes={n_episodes}...')
            env = create_env(map)
            start = timer()
            policy = run_qlearning(env, episodes=n_episodes)
            end = timer()
            mean_number_of_steps, lost_games_perc = score_frozen_lake(env, policy)
            results.append({
                'map_p': map_p,
                'map_size': map_size,
                'n_episodes': n_episodes,
                'mean_number_of_steps': mean_number_of_steps,
                'lost_games_perc': lost_games_perc,
                'time': end - start
            })

    with open(f'qlearning_stats_{map_p}.json', "wb") as f:
        f.write(json.dumps(results).encode("utf-8"))
    return results


########################################
# Value Iteration
########################################

def run_value_iteration(env, theta=0.0000001, discount_factor=0.999, max_iters=1000):
    optimal_policy, optimal_value_function, converged = value_iteration(env, theta=theta, discount_factor=discount_factor, max_iters=max_iters)
    optimal_policy_flat = np.where(optimal_policy == 1)[1]
    return optimal_policy_flat, converged


def analyze_value_iteration(map_p=0.8):
    maps = load_maps(map_p=map_p)

    map_sizes = MAP_SIZES
    max_iters_values = [100, 300, 500, 700, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    results = []

    for map_size in map_sizes:
        map = get_map(maps, map_size=map_size)
        for max_iters in max_iters_values:
            print('########################################')
            print(f'Running value iteration for map_size={map_size} and max_iters={max_iters}...')
            env = create_env(map)
            start = timer()
            policy, converged = run_value_iteration(env, max_iters=max_iters)
            end = timer()
            mean_number_of_steps, lost_games_perc = score_frozen_lake(env, policy)
            results.append({
                'map_p': map_p,
                'map_size': map_size,
                'max_iters': max_iters,
                'converged': converged,
                'mean_number_of_steps': mean_number_of_steps,
                'lost_games_perc': lost_games_perc,
                'time': end - start
            })

    with open(f'value_iteration_stats_{map_p}.json', "wb") as f:
        f.write(json.dumps(results).encode("utf-8"))
    return results


########################################
# Policy Iteration
########################################

def try_policy_iteration(lake_map):
    print('Trying frozen lake with Policy Iteration')
    env = FrozenLakeEnv(desc=lake_map)
    optimal_policy, optimal_value_function = policy_improvement(env, discount_factor=0.9999)
    optimal_policy_flat = np.where(optimal_policy == 1)[1]
    # optimal_policy_reshaped = optimal_policy_flat.reshape(4, 4)

    score_frozen_lake(env, optimal_policy_flat)


if __name__ == '__main__':
    #find_good_maps(map_p=0.95)

    #analyze_qlearning(map_p=0.9)

    analyze_value_iteration(map_p=0.8)

    print()
