import pandas as pd
import matplotlib.pyplot as plt


def size_vs_iteration():
    df = pd.read_json('forest_management_stats.json')

    stats = df[['n_states', 'pi_iters', 'vi_iters', 'ql_iters']]
    stats = stats.set_index('n_states')

    stats_pivi = stats[['pi_iters', 'vi_iters']]
    stats_pivi = stats_pivi.rename(columns={"pi_iters": "Policy Iteration", "vi_iters": "Value Iteration"})
    stats_pivi.plot(marker='o')
    plt.title('Number of iterations (Policy Iteration, Value Iteration)')
    plt.ylabel('number of iterations to find policy')
    plt.xlabel('number of states')
    plt.xticks([2, 4, 8, 12, 16])
    plt.grid()
    plt.legend()
    plt.savefig('plots/forest_size_vs_iterations_pi_vi.png')
    plt.clf()

    stats_ql = stats[['ql_iters']]
    stats_ql = stats_ql.rename(columns={"ql_iters": "Q-Learning"})
    stats_ql.plot(marker='o')
    plt.title('Number of iterations (Q-Learning)')
    plt.ylabel('number of iterations to find policy')
    plt.xlabel('number of states')
    plt.xticks([2, 4, 8, 12, 16])
    plt.grid()
    plt.legend()
    plt.savefig('plots/forest_size_vs_iterations_ql.png')
    plt.clf()


def size_vs_time():
    df = pd.read_json('forest_management_stats.json')

    stats = df[['n_states', 'pi_time', 'vi_time', 'ql_time']]
    stats = stats.set_index('n_states')

    stats_pivi = stats[['pi_time', 'vi_time']]
    stats_pivi = stats_pivi.rename(columns={"pi_time": "Policy Iteration", "vi_time": "Value Iteration"})
    stats_pivi.plot(marker='o')
    plt.title('Time to learn policy (Policy Iteration, Value Iteration)')
    plt.ylabel('time to learn policy (milliseconds)')
    plt.xlabel('number of states')
    plt.xticks([2, 4, 8, 12, 16])
    plt.grid()
    plt.legend()
    plt.savefig('plots/forest_size_vs_time_pi_vi.png')
    plt.clf()

    stats_ql = stats[['ql_time']]
    stats_ql = stats_ql.rename(columns={"ql_time": "Q-Learning"})
    stats_ql.plot(marker='o')
    plt.title('Time to learn policy (Q-Learning)')
    plt.ylabel('time to learn policy (milliseconds)')
    plt.xlabel('number of states')
    plt.xticks([2, 4, 8, 12, 16])
    plt.grid()
    plt.legend()
    plt.savefig('plots/forest_size_vs_time_ql.png')
    plt.clf()


if __name__ == '__main__':
    # size_vs_iteration()
    size_vs_time()
    print()