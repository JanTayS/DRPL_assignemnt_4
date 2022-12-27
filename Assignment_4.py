import numpy as np

epsilon = 0.9
states = 5
gamma = 0.9
pi_a = np.zeros((states,states)) # policy a
pi_b = np.zeros((states,states)) # policy b

rewards_a = np.zeros(states)
rewards_b = np.zeros(states)

rewards_a[states-1] = 1
rewards_b[0] = 0.2

Q_a = rewards_a # initialize Q values under policy a as rewards policy a
Q_b = rewards_b # initialize Q values under policy b as rewards policy b

for state in range(states):
    pi_a[state,state] = 1
    pi_b[state,0] = 1

print(pi_a)
print(pi_b)

print(rewards_a)
print(rewards_b)


