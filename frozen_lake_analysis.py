import pandas as pd
import matplotlib.pyplot as plt


# TODO: plot maps
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
    # TODO: plot qlearning in separate plot

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


if __name__ == '__main__':
    mapsize_vs_gameswon(map_p=0.8)
    mapsize_vs_iterations(map_p=0.8)
    mapsize_vs_nsteps(map_p=0.8)
    mapsize_vs_traintime(map_p=0.8)

    mapsize_vs_gameswon(map_p=0.9)
    mapsize_vs_iterations(map_p=0.9)
    mapsize_vs_nsteps(map_p=0.9)
    mapsize_vs_traintime(map_p=0.9)
