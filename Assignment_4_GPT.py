import numpy as np

# Set up the environment
n_states = 5
n_actions = 2
rewards_a = [0, 0, 0, 0, 1]  # Rewards for each state

# Initialize the Q-table
Q = np.zeros((n_states, n_actions))

print(Q)

# Set hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.1

# Train the agent
for episode in range(10000):
    # Initialize the starting state
    state = np.random.randint(n_states)
    
    # Run the episode until the terminal state is reached
    while state != n_states-1:
        # Choose an action using an epsilon-greedy strategy
        if np.random.random() < epsilon:
            action = np.random.randint(n_actions)
        else:
            action = np.argmax(Q[state])
        
        # Take the action and observe the reward and next state
        reward = rewards[state]
        if action == 0:  # action 0
            next_state = max(state-1, 0)
        elif action == 1:  # action 1
            next_state = min(state+1, n_states-1)
        
        # Update the Q-table
        Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])
        
        # Set the current state to the next state
        state = next_state