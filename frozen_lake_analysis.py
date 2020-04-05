import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gym.envs.toy_text.frozen_lake import UP, RIGHT, DOWN, LEFT
from matplotlib import colors
from matplotlib.collections import PatchCollection

from frozen_lake import load_maps, get_map, MAP_SIZES, create_env, run_policy_iteration, run_value_iteration, \
    run_qlearning
import matplotlib.patches as mpatches


# TODO: plot policies


def mapsize_vs_gameswon(map_p=0.8):
    qlearning = pd.read_json(f'qlearning_stats_{map_p}.json')
    valueit = pd.read_json(f'value_iteration_stats_{map_p}.json')
    policyit = pd.read_json(f'policy_iteration_stats_{map_p}.json')

    df = pd.DataFrame()
    df['Q Learning'] = 100 - qlearning.groupby('map_size').min()['lost_games_perc']
    df['Value Iteration'] = 100 - valueit.groupby('map_size').min()['lost_games_perc']
    df['Policy Iteration'] = 100 - policyit.groupby('map_size').min()['lost_games_perc']

    df.plot(marker='o')
    plt.title(f'Frozen Lake (p={map_p}) - Map size vs Games won')
    plt.ylabel('games won (percentage)')
    plt.xlabel('map size (length of one side)')
    plt.xticks(range(4, 36, 4))
    plt.grid()
    plt.legend()
    plt.savefig(f'plots/frozenlake_mapsize_vs_gameswon_p{map_p}.png')
    plt.clf()


def mapsize_vs_iterations(map_p=0.8):
    qlearning = pd.read_json(f'qlearning_stats_{map_p}.json')
    valueit = pd.read_json(f'value_iteration_stats_{map_p}.json')
    policyit = pd.read_json(f'policy_iteration_stats_{map_p}.json')

    qlearning.loc[qlearning['lost_games_perc'] == 100, 'n_episodes'] = 300000

    qlearning_idx = qlearning.groupby('map_size').idxmin()['lost_games_perc']
    qlearning_mapsize_iters = qlearning.iloc[qlearning_idx]['n_episodes']

    valueit_idx = valueit.groupby('map_size').idxmin()['lost_games_perc']
    valueit_mapsize_iters = valueit.iloc[valueit_idx]['max_iters']

    policyit_idx = policyit.groupby('map_size').idxmin()['lost_games_perc']
    policyit_mapsize_iters = policyit.iloc[policyit_idx]['max_iters']

    df = pd.DataFrame(index=[4, 8, 12, 16, 20, 24, 28, 32])
    df['Value Iteration'] = valueit_mapsize_iters.tolist()
    df['Policy Iteration'] = policyit_mapsize_iters.tolist()

    df.plot(marker='o')
    plt.title(f'Frozen Lake (p={map_p}) - Map size vs Iterations')
    plt.ylabel('train iterations')
    plt.xlabel('map size (length of one side)')
    plt.xticks(range(4, 36, 4))
    plt.grid()
    plt.legend()
    plt.savefig(f'plots/frozenlake_mapsize_vs_iterations_p{map_p}.png')
    plt.clf()

    df = pd.DataFrame(index=[4, 8, 12, 16, 20, 24, 28, 32])
    df['Q Learning'] = qlearning_mapsize_iters.tolist()

    df.plot(marker='o')
    plt.title(f'Frozen Lake (p={map_p}) - Map size vs Iterations (Q Learning)')
    plt.ylabel('train iterations')
    plt.xlabel('map size (length of one side)')
    plt.xticks(range(4, 36, 4))
    plt.grid()
    plt.legend()
    plt.savefig(f'plots/frozenlake_mapsize_vs_iterations_qlearning_p{map_p}.png')
    plt.clf()


def mapsize_vs_nsteps(map_p=0.8):
    qlearning = pd.read_json(f'qlearning_stats_{map_p}.json')
    valueit = pd.read_json(f'value_iteration_stats_{map_p}.json')
    policyit = pd.read_json(f'policy_iteration_stats_{map_p}.json')

    qlearning.loc[qlearning['lost_games_perc'] == 100, 'mean_number_of_steps'] = 999

    qlearning_idx = qlearning.groupby('map_size').idxmin()['lost_games_perc']
    qlearning_mapsize_nsteps = qlearning.iloc[qlearning_idx]['mean_number_of_steps']

    valueit_idx = valueit.groupby('map_size').idxmin()['lost_games_perc']
    valueit_mapsize_nsteps = valueit.iloc[valueit_idx]['mean_number_of_steps']

    policyit_idx = policyit.groupby('map_size').idxmin()['lost_games_perc']
    policyit_mapsize_nsteps = policyit.iloc[policyit_idx]['mean_number_of_steps']

    df = pd.DataFrame(index=[4, 8, 12, 16, 20, 24, 28, 32])
    df['Q Learning'] = qlearning_mapsize_nsteps.tolist()
    df['Value Iteration'] = valueit_mapsize_nsteps.tolist()
    df['Policy Iteration'] = policyit_mapsize_nsteps.tolist()

    df.plot(marker='o')
    plt.title(f'Frozen Lake (p={map_p}) - Map size vs Mean number of steps to goal')
    plt.ylabel('mean number of steps to reach goal')
    plt.xlabel('map size (length of one side)')
    plt.xticks(range(4, 36, 4))
    plt.grid()
    plt.legend()
    plt.savefig(f'plots/frozenlake_mapsize_vs_nsteps_p{map_p}.png')
    plt.clf()


def mapsize_vs_traintime(map_p=0.8):
    qlearning = pd.read_json(f'qlearning_stats_{map_p}.json')
    valueit = pd.read_json(f'value_iteration_stats_{map_p}.json')
    policyit = pd.read_json(f'policy_iteration_stats_{map_p}.json')

    # qlearning.loc[qlearning['lost_games_perc'] == 100, 'mean_number_of_steps'] = 999

    qlearning_idx = qlearning.groupby('map_size').idxmin()['lost_games_perc']
    qlearning_mapsize_time = qlearning.iloc[qlearning_idx]['time']

    valueit_idx = valueit.groupby('map_size').idxmin()['lost_games_perc']
    valueit_mapsize_time = valueit.iloc[valueit_idx]['time']

    policyit_idx = policyit.groupby('map_size').idxmin()['lost_games_perc']
    policyit_mapsize_time = policyit.iloc[policyit_idx]['time']

    df = pd.DataFrame(index=[4, 8, 12, 16, 20, 24, 28, 32])
    df['Q Learning'] = qlearning_mapsize_time.tolist()
    df['Value Iteration'] = valueit_mapsize_time.tolist()
    df['Policy Iteration'] = policyit_mapsize_time.tolist()

    df.plot(marker='o')
    plt.title(f'Frozen Lake (p={map_p}) - Map size vs Time required to train')
    plt.ylabel('time required to train (seconds)')
    plt.xlabel('map size (length of one side)')
    plt.xticks(range(4, 36, 4))
    plt.grid()
    plt.legend()
    plt.savefig(f'plots/frozenlake_mapsize_vs_traintime_p{map_p}.png')
    plt.clf()

    print()


def draw_map(map_p=0.8, map_size=32):
    maps = load_maps(map_p=map_p)
    map = get_map(maps, map_size=map_size)

    def to_scalar(x):
        if x == 'S': return 0
        if x == 'F': return 1
        if x == 'H': return 2
        if x == 'G': return 3

    def to_scalar_list(row):
        return [to_scalar(elem) for elem in row]

    map_2d = [to_scalar_list(x) for x in [list(row) for row in map]]

    fig, ax = plt.subplots(figsize=(15, 15))
    cmap = colors.ListedColormap(['green', 'lightblue', 'darkblue', 'red'])
    plt.pcolor(map_2d, edgecolors='k', cmap=cmap)
    plt.title(f'Frozen Lake map ({map_size}x{map_size}, p={map_p})')
    plt.xticks([])
    plt.yticks([])
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'plots/maps/frozenlake_map_{map_size}x{map_size}_p{map_p}.png')
    plt.clf()


def draw_all_maps(map_p=0.8):
    for map_size in MAP_SIZES:
        draw_map(map_p, map_size)


def draw_policy(algo='value_iteration', map_p=0.8, map_size=8):
    maps = load_maps(map_p=map_p)
    map = get_map(maps, map_size=map_size)
    env = create_env(map)

    # Get policy
    policy = None
    if algo == 'value_iteration':
        policy, _ = run_value_iteration(env, max_iters=5000)
    if algo == 'policy_iteration':
        policy, _, _ = run_policy_iteration(env, max_iters=5000)
    if algo == 'qlearning':
        policy = run_qlearning(env, episodes=200000)

    # Draw Policy
    def arrow_down(x, y, arrow_size=0.6):
        return mpatches.Arrow(x + 0.5, y + 0.2, 0, arrow_size)

    def arrow_up(x, y, arrow_size=0.6):
        return mpatches.Arrow(x + 0.5, y + arrow_size + 0.2, 0, -arrow_size)

    def arrow_right(x, y, arrow_size=0.6):
        return mpatches.Arrow(x + 0.2, y + 0.5, arrow_size, 0)

    def arrow_left(x, y, arrow_size=0.6):
        return mpatches.Arrow(x + arrow_size + 0.2, y + 0.5, -arrow_size, 0)

    policy_2d = np.resize(policy, (map_size, map_size))
    policy_2d_list = list(policy_2d.tolist())
    patches = []
    for row in range(len(policy_2d_list)):
        for col in range(len(policy_2d_list[0])):
            action = policy_2d_list[row][col]
            arrow = None
            if action == UP:
                arrow = arrow_up(col, row)
            if action == RIGHT:
                arrow = arrow_right(col, row)
            if action == DOWN:
                arrow = arrow_down(col, row)
            if action == LEFT:
                arrow = arrow_left(col, row)
            patches.append(arrow)


    # Draw map
    def to_scalar(x):
        if x == 'S': return 0
        if x == 'F': return 1
        if x == 'H': return 2
        if x == 'G': return 3

    def to_scalar_list(row):
        return [to_scalar(elem) for elem in row]

    map_2d = [to_scalar_list(x) for x in [list(row) for row in map]]

    fig, ax = plt.subplots(figsize=(15,15))
    cmap = colors.ListedColormap(['green', 'lightblue', 'darkblue', 'red'])
    ax.pcolor(map_2d, edgecolors='k', cmap=cmap)
    plt.title(f'Policy found by {algo} on Frozen Lake ({map_size}x{map_size} map, p={map_p})')
    plt.xticks([])
    plt.yticks([])
    collection = PatchCollection(patches, color='white')
    ax.add_collection(collection)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'plots/policies/frozenlake_{algo}_{map_size}x{map_size}_p{map_p}.png')
    plt.clf()


if __name__ == '__main__':
    # mapsize_vs_gameswon(map_p=0.8)
    # mapsize_vs_iterations(map_p=0.8)
    # mapsize_vs_nsteps(map_p=0.8)
    # mapsize_vs_traintime(map_p=0.8)

    # mapsize_vs_gameswon(map_p=0.9)
    # mapsize_vs_iterations(map_p=0.9)
    # mapsize_vs_nsteps(map_p=0.9)
    # mapsize_vs_traintime(map_p=0.9)

    # draw_all_maps(map_p=0.8)
    # draw_all_maps(map_p=0.9)

    # draw_policy(algo='value_iteration', map_p=0.8, map_size=32)
    # draw_policy(algo='policy_iteration', map_p=0.8, map_size=32)
    # draw_policy(algo='qlearning', map_p=0.8, map_size=32)

    draw_policy(algo='value_iteration', map_p=0.8, map_size=8)
    draw_policy(algo='policy_iteration', map_p=0.8, map_size=8)
    draw_policy(algo='qlearning', map_p=0.8, map_size=8)
