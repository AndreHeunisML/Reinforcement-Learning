

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np
import gym


def calculate_mc_returns(rewards, gamma=0.99):
    n = len(rewards)
    returns = np.zeros(n)
    returns[-1] = rewards[-1]

    for i in range(n-2, -1, -1):
        returns[i] = rewards[i] + gamma * returns[i+1]

    return returns


def train(total_batches=50000):
    """
    Using Monte Carlo to train an agent using vanilla policy gradient. Since we are doing Monte Carlo we do not need a
    value function approximator (we have all the true values because we play each episode to the end).

    Since we dont have a value function approximator, we use the return instead of the advantage in the cost function

    :param total_batches:
    :return:
    """

    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    action_space = env.action_space.n

    states = tf.placeholder(dtype=tf.float32, shape=(None, state_size))
    actions = tf.placeholder(dtype=tf.int32, shape=(None, action_space))
    returns = tf.placeholder(dtype=tf.float32, shape=(1,))

    l1 = fully_connected(inputs=states, num_outputs=32, activation_fn=tf.tanh)
    l2 = fully_connected(inputs=l1, num_outputs=32, activation_fn=tf.tanh)
    logits = fully_connected(inputs=l2, num_outputs=action_space, activation_fn=tf.tanh)

    actions_op = tf.squeeze(tf.multinomial(logits=logits, num_samples=1))

    log_probs = tf.reduce_sum(tf.one_hot(actions, depth=action_space) * tf.nn.log_softmax(logits), axis=1)
    loss = -1 * tf.reduce_mean(returns * log_probs)
    optim = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optim.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for b in range(total_batches):
        current_state, reward, done, _ = env.reset()
        done = False

        ep_states = []
        ep_rewards = []
        ep_actions = []

        # Run an episode to the end
        while not done:
            action = sess.run(actions_op, feed_dict={states: current_state})[0]
            current_state, reward, done, _ = env.step(action)

            ep_states.append(current_state)
            ep_rewards.append(reward)
            ep_actions.append(action)

        ep_returns = calculate_mc_returns(ep_rewards)

        sess.run(train_op, feed_dict={
            states: ep_states,
            actions: ep_actions,
            returns: ep_returns
        })

if __name__ == '__main__':
    train()
