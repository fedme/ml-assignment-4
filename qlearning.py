# From https://www.kaggle.com/angps95/intro-to-reinforcement-learning-with-openai-gym

import numpy as np


def q_learning(env, alpha, gamma, epsilon, episodes):
    """Q Learning Algorithm with epsilon greedy

    Args:
        env: Environment
        alpha: Learning Rate --> Extent to which our Q-values are being updated in every iteration.
        gamma: Discount Rate --> How much importance we want to give to future rewards
        epsilon: Probability of selecting random action instead of the 'optimal' action
        episodes: No. of episodes to train on

    Returns:
        Q-learning Trained policy

    """
    """Training the agent"""

    # For plotting metrics
    all_epochs = []
    all_penalties = []

    # Initialize Q table of 500 x 6 size (500 states and 6 actions) with all zeroes
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    for i in range(1, episodes + 1):
        state = env.reset()

        epochs, penalties, reward, = 0, 0, 0
        done = False

        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore action space randomly
            else:
                action = np.argmax(q_table[state])  # Exploit learned values by choosing optimal values

            next_state, reward, done, info = env.step(action)

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward == -10:
                penalties += 1

            state = next_state
            epochs += 1

        if i % 100 == 0:
            print(f"Episode: {i}")
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA

    for state in range(env.nS):  # for each states
        best_act = np.argmax(q_table[state])  # find best action
        policy[state] = np.eye(env.nA)[best_act]  # update

    print("Training finished.\n")
    return policy, q_table