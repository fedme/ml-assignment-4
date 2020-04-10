import json

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
#
#  This function is used to generate a transition probability
#     (``A`` × ``S`` × ``S``) array ``P`` and a reward (``S`` × ``A``) matrix
#     ``R`` that model the following problem. A forest is managed by two actions:
#     'Wait' and 'Cut'. An action is decided each year with first the objective
#     to maintain an old forest for wildlife and second to make money selling cut
#     wood. Each year there is a probability ``p`` that a fire burns the forest.
#
#     Here is how the problem is modelled.
#     Let {0, 1 . . . ``S``-1 } be the states of the forest, with ``S``-1 being
#     the oldest. Let 'Wait' be action 0 and 'Cut' be action 1.
#     After a fire, the forest is in the youngest state, that is state 0.


def compute_policy_distance(policy1, policy2):
    assert len(policy1) == len(policy2)
    distance = 0
    for i in range(len(policy1)):
        if policy1[i] != policy2[i]:
            distance = distance + 1
    return distance



def forest_management():
    results = []

    n_states_values = [2, 4, 8, 12, 16]
    for n_states in n_states_values:
        result = {'n_states': n_states}

        # PROBLEM
        P, R = example.forest(S=n_states, r1=4, r2=2, p=0.3)

        # Policy Iteration
        print(f'[n_states={n_states}] Running PI...')
        pi = PolicyIteration(transitions=P, reward=R, gamma=0.95)
        pi.run()
        result['pi_iters'] = pi.iter
        result['pi_time'] = pi.time * 1000
        result['pi_policy'] = pi.policy

        # Value Iteration
        print(f'[n_states={n_states}] Running VI...')
        vi = ValueIteration(transitions=P, reward=R, gamma=0.95)
        vi.run()
        result['vi_iters'] = vi.iter
        result['vi_time'] = vi.time * 1000
        result['vi_policy'] = vi.policy

        result['pi_vi_policies_same'] = pi.policy == vi.policy

        # Q-Learning with Gridsearch
        ql_iters_values = [10000, 25000, 50000, 75000, 100000]
        alpha_values = [0.1, 0.2, 0.4]
        alpha_decay_values = [0.99, 0.9, 0.8, 0.6]
        epsilon_decay_values = [0.99, 0.9, 0.8, 0.6]

        grid = [(ql_iters, alpha, alpha_decay, epsilon_decay) for ql_iters in ql_iters_values
                for alpha in alpha_values
                for alpha_decay in alpha_decay_values
                for epsilon_decay in epsilon_decay_values]

        min_policy_distance = float('inf')
        iteration = 0

        for (ql_iters, alpha, alpha_decay, epsilon_decay) in grid:
            iteration = iteration + 1
            print(f'[n_states={n_states}] Running QL with gridsearch ({iteration}/{len(grid)})...')
            ql = QLearning(transitions=P, reward=R, gamma=0.95,
                           alpha=alpha, alpha_decay=alpha_decay, alpha_min=0.001,
                           epsilon=1.0, epsilon_min=0.1, epsilon_decay=epsilon_decay,
                           n_iter=ql_iters)
            ql.run()
            policy_distance = compute_policy_distance(ql.policy, pi.policy)

            if policy_distance < min_policy_distance:
                min_policy_distance = policy_distance
                result['ql_iters'] = ql_iters
                result['ql_time'] = ql.time * 1000
                result['ql_policy'] = ql.policy
                result['ql_policy_distance'] = compute_policy_distance(ql.policy, pi.policy)
                result['ql_alpha'] = alpha
                result['ql_alpha_decay'] = alpha_decay
                result['ql_epsilon_decay'] = epsilon_decay

        # Save results
        results.append(result)

    with open(f'forest_management_stats.json', "wb") as f:
        f.write(json.dumps(results).encode("utf-8"))


if __name__ == '__main__':
    forest_management()
    print()