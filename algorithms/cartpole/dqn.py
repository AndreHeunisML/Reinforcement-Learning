
import numpy as np
import tensorflow as tf
import gym
from pylab import *

from algorithms.utils import calc_running_avg


class DQN:

    def __init__(self, sess, state_dim, action_dim, scope, target_model=None):
        self.sess = sess
        self.action_dim = action_dim
        self.gamma = 0.99
        self.exp_buf_size = 1000
        self.exp_buf = []
        self.min_exp = 50
        self.batch_size = 32
        self.target_model = target_model
        self.scope = scope

        with tf.variable_scope(scope):

            self.states = tf.placeholder(tf.float32, shape=(None, state_dim))
            self.actions = tf.placeholder(tf.int32, shape=(None,))
            self.target_returns = tf.placeholder(tf.float32, shape=(None,))

            l1 = tf.contrib.layers.fully_connected(inputs=self.states, num_outputs=32, activation_fn=tf.nn.tanh)
            self.logits = tf.contrib.layers.fully_connected(inputs=l1, num_outputs=action_dim, activation_fn=None)

            self.action_op = tf.argmax(self.logits)
            returns = tf.reduce_sum(tf.one_hot(self.actions, depth=action_dim) * self.logits, axis=1)
            loss = tf.reduce_sum(tf.square(self.target_returns - returns))

            optim = tf.train.AdamOptimizer(learning_rate=0.01)
            self.train_op = optim.minimize(loss=loss)

    def predict_logits(self, states):
        return self.sess.run(self.logits, feed_dict={self.states: np.atleast_2d(states)})

    def sample_action(self, states, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.action_dim)
        else:
            return np.argmax(self.predict_logits(states), axis=1)[0]

    def add_to_exp_buffer(self, state, action, reward, next_state, done):
        self.exp_buf.append((state, action, reward, next_state, done))

    def train(self):

        if len(self.exp_buf) < self.min_exp:
            return

        batch_idx = np.random.choice(len(self.exp_buf), size=self.batch_size, replace=False)
        batch = [self.exp_buf[i] for i in batch_idx]

        states = [s[0] for s in batch]
        actions = [s[1] for s in batch]
        rewards = [s[2] for s in batch]
        next_states = [s[3] for s in batch]
        dones = [s[4] for s in batch]

        # Use target model to get the target returns
        if self.target_model is not None:
            target_q = self.target_model.predict_logits(next_states)
        else:
            target_q = self.predict_logits(next_states)

        target_return = [r + self.gamma * max(tq) if not d else r for r, tq, d in zip(rewards, target_q, dones)]

        self.sess.run(self.train_op, feed_dict={
            self.states: states,
            self.actions: actions,
            self.target_returns: target_return})

    def copy_params_from(self, model):
        mine = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        mine = sorted(mine, key=lambda v: v.name)
        theirs = [t for t in tf.trainable_variables() if t.name.startswith(model.scope)]
        theirs = sorted(theirs, key=lambda v: v.name)

        ops = []
        for p, q in zip(mine, theirs):
            actual = self.sess.run(q)
            op = p.assign(actual)
            ops.append(op)

        self.sess.run(ops)


def run_training_episodes():

    env = gym.make('CartPole-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    num_episodes = 350
    target_model_copy_steps = 50
    episode_rewards = np.zeros(num_episodes)

    sess = tf.Session()

    target = DQN(sess, state_dim, action_dim, scope='target')
    model = DQN(sess, state_dim, action_dim, scope='model', target_model=target)

    sess.run(tf.global_variables_initializer())

    total_steps = 0
    for n in range(num_episodes):

        current_state = env.reset()
        done = False
        ep_reward = 0
        eps = 1.0 / np.sqrt(n + 1)

        while not done:
            action = model.sample_action(current_state, eps)
            prev_state = current_state

            current_state, reward, done, _ = env.step(action)
            ep_reward += reward

            # add to experience
            model.add_to_exp_buffer(prev_state, action, reward, current_state, done)

            # train if enough exp has been gathered
            model.train()
            total_steps += 1

            # update target network if necessary
            if target is not None and (total_steps + 1) % target_model_copy_steps == 0:
                print('updating target network params')
                target.copy_params_from(model)

        episode_rewards[n] = ep_reward

        if n % 10 == 0:
            print('ep: {} \t avg reward (prev 100): {:.3f}'.format(
                n,
                np.mean(episode_rewards[max([0, n-100]):n+1])), )

    return episode_rewards


if __name__ == '__main__':
    train_rewards = run_training_episodes()

    running_avg = calc_running_avg(train_rewards)

    figure()
    plot(train_rewards)
    plot(running_avg)
    show()



