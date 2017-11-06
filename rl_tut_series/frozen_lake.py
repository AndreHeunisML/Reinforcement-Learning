import gym
import numpy as np
from matplotlib.pylab import figure, plot, show

env = gym.make('FrozenLake-v0')

# Initialize table with all zeros
print 'observations'
print env.observation_space
print env.observation_space.n
print
print 'actions'
print env.action_space
print env.action_space.n

q = np.zeros([env.observation_space.n, env.action_space.n])

# Set learning parameters
lr = .8                 # learning rate
y = .95                 # discounting factor
num_episodes = 20000

# create lists to contain total rewards and steps per episode
rList = []
time_to_success = []
for i in range(num_episodes):

    # print '******* episode {} *********'.format(i)

    # Reset environment and get first new observation
    s = env.reset()
    r_all = 0
    d = False
    j = 0

    # The Q-Table learning algorithm
    while j < 99:
        j += 1
        # Choose an action by greedily (with noise) picking from Q table
        a = np.argmax(q[s, :] + np.random.randn(1, env.action_space.n)*(1./(i+1)))

        # Get new state and reward from environment. observation, reward, done, info
        s1, r, d, _ = env.step(a)

        if r > 0:
            print
            print env.render()
            print 'state: {}'.format(s)
            print 'action: {}'.format(a)
            print 'reward: {}'.format(r)
            print 'new state: {}'.format(s1)
            print 'future reward: {}'.format(y * np.max(q[s1, :]) - q[s, a])

        # Update Q-Table with new knowledge
        # Q-table:          value of being in a certain state and taking a certain action.
        # r:                reward for current action in current state.
        # y:                discount factor.
        # np.max(Q[s1, :])  best possible action to take in the next state
        q[s, a] = q[s, a] + lr * (r + y * np.max(q[s1, :]) - q[s, a])
        r_all += r
        s = s1

        if d:
            if r > 0:
                time_to_success.append(j)
            break

    rList.append(r_all)

print "Score over time: " + str(sum(rList)/num_episodes)

figure()
plot(time_to_success)
show()
