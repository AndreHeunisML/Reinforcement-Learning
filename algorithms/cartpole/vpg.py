

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np
import gym
from pylab import *

from algorithms.utils import calc_running_avg


def calculate_mc_returns(rewards, gamma=0.99):
    n = len(rewards)

    if n == 0:
        return []

    returns = np.zeros(n)
    returns[-1] = rewards[-1]

    for i in range(n-2, -1, -1):
        returns[i] = rewards[i] + gamma * returns[i+1]

    return returns


def train(batch_size=5000, total_batches=50):
    """
    Using Monte Carlo to train an agent using vanilla policy gradient. Since we are doing Monte Carlo we do not need a
    value function approximator (we have all the true values because we play each episode to the end).

    Since we dont have a value function approximator, we use the return instead of the advantage in the cost function

    :param total_batches:
    :return:
    """

    env = gym.make('CartPole-v0')
    # env._max_episode_steps = 1000
    state_size = env.observation_space.shape[0]
    action_space = env.action_space.n

    print('state size: {}'.format(state_size))
    print('action size: {}'.format(action_space))

    states = tf.placeholder(dtype=tf.float32, shape=(None, state_size))
    actions = tf.placeholder(dtype=tf.int32, shape=(None,))
    returns = tf.placeholder(dtype=tf.float32, shape=(None,))

    l1 = fully_connected(inputs=states, num_outputs=32, activation_fn=tf.nn.tanh)
    #l2 = fully_connected(inputs=l1, num_outputs=12, activation_fn=tf.nn.relu)
    logits = fully_connected(inputs=l1, num_outputs=action_space, activation_fn=None)

    actions_op = tf.squeeze(tf.multinomial(logits=logits, num_samples=1))

    log_probs = tf.reduce_sum(tf.one_hot(actions, depth=action_space) * tf.nn.log_softmax(logits), axis=1)
    loss = -1 * tf.reduce_mean(returns * log_probs)
    optim = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optim.minimize(loss=loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    all_episode_rewards = []
    all_batch_loss = np.zeros(total_batches)
    episode_lengths = []
    total_steps_taken = 0

    for b in range(total_batches):
        done = True

        # for training
        batch_states = []
        batch_returns = []
        batch_actions = []
        episode_rewards = []

        # for evaluation
        total_ep_reward = 0

        # Run an episode to the end
        for bi in range(batch_size):
            if done:
                current_state = env.reset()
                batch_returns += list(calculate_mc_returns(episode_rewards))

                episode_lengths.append(len(episode_rewards))
                episode_rewards = []
                all_episode_rewards.append(total_ep_reward)
                total_ep_reward = 0

            batch_states.append(current_state)
            action = sess.run(actions_op, feed_dict={states: current_state.reshape((1, state_size))})
            current_state, reward, done, _ = env.step(action)
            total_steps_taken += 1

            total_ep_reward += reward

            episode_rewards.append(reward)
            batch_actions.append(action)

        # append returns from the final episode
        batch_returns += list(calculate_mc_returns(episode_rewards))

        batch_advs = np.array(batch_returns)
        batch_advs = (batch_advs - np.mean(batch_advs)) / (np.std(batch_advs) + 1e-8)

        _, tloss = sess.run((train_op, loss), feed_dict={
            states: batch_states,
            actions: batch_actions,
            returns: batch_advs
        })

        all_batch_loss[b] = tloss

        if b % 1 == 0 and b > 0:
            print('batch: {:.3f} \t avg reward (prev 100): {:.3f} \t avg ep length (prev 100): {:.3f}'.format(
                b,
                np.median(all_episode_rewards[-100:]),
                np.mean(episode_lengths[-100:])))

    print("Total episodes: ", len(episode_lengths))
    print("Total steps taken: ", total_steps_taken)

    return np.array(all_episode_rewards), np.array(all_batch_loss)


if __name__ == '__main__':
    train_rewards, train_loss = train()

    total_episodes = len(train_rewards)

    running_avg = calc_running_avg(train_rewards)

    figure(figsize=(16, 9))
    subplot(211)
    plot(train_rewards)
    plot(running_avg)
    subplot(212)
    plot(train_loss)
    show()
