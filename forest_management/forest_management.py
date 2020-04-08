from hiive.mdptoolbox import example
from hiive.mdptoolbox.mdp import PolicyIteration, ValueIteration, QLearning


# Forest Management Parameters
#     ---------
#     S : int, optional
#         The number of states, which should be an integer greater than 1.
#         Default: 3.
#     r1 : float, optional
#         The reward when the forest is in its oldest state and action 'Wait' is
#         performed. Default: 4.
#     r2 : float, optional
#         The reward when the forest is in its oldest state and action 'Cut' is
#         performed. Default: 2.
#     p : float, optional
#         The probability of wild fire occurence, in the range ]0, 1[. Default:
#         0.1.
#     is_sparse : bool, optional
#         If True, then the probability transition matrices will be returned in
#         sparse format, otherwise they will be in dense format. Default: False.


def throw_away():

    n_states_values = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    results = []

    for n_states in n_states_values:
        print(f'Running forest management with n_states={n_states}.')

        P, R = example.forest(S=n_states, r1=4, r2=2, p=0.1)

        pi = PolicyIteration(transitions=P, reward=R, gamma=0.95)
        pi.run()

        vi = ValueIteration(transitions=P, reward=R, gamma=0.95)
        vi.run()

        ql = QLearning(transitions=P, reward=R, gamma=0.95, n_iter=100000)
        ql.run()

        results.append({
            'n_states': n_states,
            'pi_iters': pi.iter,
            'pi_time': pi.time * 1000,
            'pi_policy': pi.policy,
            'vi_iters': vi.iter,
            'vi_time': vi.time * 1000,
            'vi_policy': vi.policy,
            'ql_iters': ql.max_iter,
            'ql_time': ql.time * 1000,
            'ql_policy': ql.policy,
            'policies_same': pi.policy == vi.policy == ql.policy,
            'pi_vi_policies_same': pi.policy == vi.policy
        })

    print()


if __name__ == '__main__':
    throw_away()
    print()