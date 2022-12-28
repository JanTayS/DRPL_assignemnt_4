import numpy as np

# Set up the environment
n_states = 5
n_actions = 2

rewards = np.zeros((n_states,n_actions))
rewards[n_states-1,0] = 1
rewards[0,1] = 0.2


# Initialize the Q-table
Q = np.zeros((n_states, n_actions))

# Set hyperparameters
epsilon = 0.1 # greedy value
gamma = 0.9 # discount factor
alpha = 1 # learning rate


# creates policy tables
pi_a = np.zeros((n_states,n_states)) # policy a
pi_b = np.zeros((n_states,n_states)) # policy b

for state in range(n_states): # initialize values for the policies
    pi_a[state,state] = 1
    pi_b[state,0] = 1


episodes = 10

for episode in range(episodes):
    state = np.random.randint(n_states)
    for run in range(1000):
        if np.random.random() < epsilon:
            action = np.random.randint(n_actions)  # Explore with probability epsilon
        else:
            action = np.argmax(Q[state])  # Choose the action with the highest Q-value with probability (1-epsilon)

    reward = rewards[state,action]

    if action == 0:
        if state == 4:
            next_state = 0
        else:
            next_state = state + 1
    elif action == 1:
        next_state = 0

    Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])

    state = next_state

print(Q)