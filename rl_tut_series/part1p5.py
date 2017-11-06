# Adapted from https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-1-5-contextual-bandits-bff01d1aad9c

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


class ContextualBandit:
    def __init__(self):
        """
        Initialise 3 4-armed bandits. Each require different actions (pulling of arms) to achieve the best result.
        """
        self.state = 0

        # List out our bandits. Currently arms 4, 2, and 1 (respectively) are the most optimal.
        self.bandits = np.array([[0.2, 0, -0.0, -5], [0.1, -5, 1, 0.25], [-5, 5, 5, 5]])
        self.num_bandits = self.bandits.shape[0]
        self.num_actions = self.bandits.shape[1]

    def get_bandit(self):
        """
        # Returns a random state (bandit) for each episode.
        """
        self.state = np.random.randint(0, len(self.bandits))
        return self.state

    def pull_arm(self, action):
        """

        :param action:
        :return:
        """
        # Get a random number.
        bandit = self.bandits[self.state, action]
        result = np.random.randn(1)
        if result > bandit:
            # return a positive reward.
            return 1
        else:
            # return a negative reward.
            return -1


class Agent:
    def __init__(self, lr, s_size, a_size):
        """


        :param lr:      Learning Rate.
        :param s_size:  Number of possible states.
        :param a_size:  umber of possible actions.
        """
        # These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in = tf.placeholder(shape=[1], dtype=tf.int32)
        state_in_oh = slim.one_hot_encoding(self.state_in, s_size)
        output = slim.fully_connected(state_in_oh,                  # TODO: what is this slim for?
                                      a_size,
                                      biases_initializer=None,
                                      activation_fn=tf.nn.sigmoid,
                                      weights_initializer=tf.ones_initializer())
        self.output = tf.reshape(output, [-1])
        self.chosen_action = tf.argmax(self.output, 0)

        # The next six lines establish the training procedure. We feed the reward and chosen action into the network
        # to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
        self.responsible_weight = tf.slice(self.output, self.action_holder, [1])
        self.loss = -(tf.log(self.responsible_weight) * self.reward_holder)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        self.update = optimizer.minimize(self.loss)

if __name__ == '__main__':
    # Clear the Tensorflow graph.
    tf.reset_default_graph()

    # Load the bandits.
    cBandit = ContextualBandit()

    # Load the agent.
    myAgent = Agent(lr=0.001, s_size=cBandit.num_bandits, a_size=cBandit.num_actions)
    weights = tf.trainable_variables()[0] # The weights we will evaluate to look into the network.

    # Set total number of episodes to train agent on.
    total_episodes = 10000

    # Set scoreboard for bandits to 0.
    total_reward = np.zeros([cBandit.num_bandits, cBandit.num_actions])

    # Set the chance of taking a random action.
    e = 0.1

    init = tf.initialize_all_variables()

    # Launch the tensorflow graph
    with tf.Session() as sess:
        sess.run(init)
        i = 0
        while i < total_episodes:
            # Get a random state from the environment.
            s = cBandit.get_bandit()

            # Choose either a random action or one from our network.
            if np.random.rand(1) < e:
                action = np.random.randint(cBandit.num_actions)
            else:
                action = sess.run(myAgent.chosen_action, feed_dict={myAgent.state_in: [s]})

            # Get our reward for taking an action given a bandit.
            reward = cBandit.pull_arm(action)

            # Update the network.
            feed_dict = {myAgent.reward_holder: [reward], myAgent.action_holder: [action], myAgent.state_in: [s]}
            _, ww = sess.run([myAgent.update, weights], feed_dict=feed_dict)

            # Update our running tally of scores.
            total_reward[s, action] += reward
            if i % 500 == 0:
                print "Mean reward for each of the " + str(cBandit.num_bandits) + " bandits: " + str(np.mean(total_reward, axis=1))
            i += 1
    for a in range(cBandit.num_bandits):
        print "The agent thinks action " + str(np.argmax(ww[a])+1) + " for bandit " + str(a+1) + " is the most promising...."
        if np.argmax(ww[a]) == np.argmin(cBandit.bandits[a]):
            print "...and it was right!"
        else:
            print "...and it was wrong!"
