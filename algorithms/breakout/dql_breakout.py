
import gym
import numpy as np
from pylab import *
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, conv2d, flatten
import itertools

from udemy_rl_2.image_utils import downsample_image, update_state


class DQN:

    def __init__(
            self,
            state_shape,
            num_actions,
            gamma,
            learning_rate,
            exp_buf_size,
            min_train_exp,
            batch_size,
            scope):
        """


        :param state_shape:     Tuple containing the shape of a single experience as (im_height, im_width, channels)
        :param num_actions:
        :param gamma:
        :param learning_rate:
        :param exp_buf_size:
        :param min_train_exp:
        :param batch_size:
        :param scope:
        """

        self.experience_buffer = []
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.num_actions = num_actions
        self.exp_buf_size = exp_buf_size
        self.min_train_exp = min_train_exp
        self.batch_size = batch_size
        self.scope = scope
        self.update_t = update_t
        self.session = None

        with tf.variable_scope(scope):
            self.current_state = tf.placeholder(tf.float32, [None, state_shape[0], state_shape[1], state_shape[2]])
            self.action = tf.placeholder(tf.int32, [None])
            self.target_return = tf.placeholder(tf.float32, [None])

            # Hidden Layers
            c1 = conv2d(inputs=self.current_state, num_outputs=32, kernel_size=8)
            c2 = conv2d(inputs=c1, num_outputs=64, kernel_size=4)

            # Output Layer
            c2_flat = flatten(c2)
            output = fully_connected(inputs=c2_flat, num_outputs=num_actions, activation_fn=None,
                                     weights_initializer=tf.initializers.random_normal())
            self.predict_op = output

            selected_action_values = tf.reduce_sum(
                output * tf.one_hot(self.action, num_actions),
                reduction_indices=[1]
            )

            self.cost = tf.reduce_sum(tf.square(self.target_return - selected_action_values))
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    def set_session(self, sess):
        self.session = sess

    def predict_q(self, state_batch):
        return self.session.run(self.predict_op, feed_dict={self.current_state: state_batch})

    def sample_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.predict_q(state))

    def update(self, target_model):
        """

        :return:
        """
        # Sample a batch from the exp buffer
        samples = random.sample(self.experience_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*samples))

        # Calculate targets
        next_Qs = target_model.predict(next_states)
        next_Q = np.amax(next_Qs, axis=1)
        targets = rewards + np.invert(dones).astype(np.float32) * self.gamma * next_Q

        # Update model
        feed_dict = {self.current_state: states, self.action: actions, self.target_return: targets}
        self.session.run(self.train_op, feed_dict=feed_dict)

    def copy_from(self, other):
        mine = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        mine = sorted(mine, key=lambda v: v.name)
        theirs = [t for t in tf.trainable_variables() if t.name.startswith(other.scope)]
        theirs = sorted(theirs, key=lambda v: v.name)

        ops = []
        for p, q in zip(mine, theirs):
            actual = self.session.run(q)
            op = p.assign(actual)
            ops.append(op)

        self.session.run(ops)

    def add_experience(self, exp):
        """

        :param exp:     Experience tuple (s, a, r, s')
        :return:
        """

        if len(self.experience_buffer) > self.exp_buf_size:
            self.experience_buffer.pop(0)

        self.experience_buffer.append(exp)


def play_episode(env, model, target, epsilon, t_total, update_t):
    """

    :param env:
    :param model:
    :param target:
    :param epsilon:
    :return:
    """

    obs = env.reset()
    current_state = update_state(np.stack([obs] * 4, axis=0), obs)
    total_ep_reward = 0

    done = False
    while not done:
        # Check if the target should be updated
        if t_total > update_t:
            t_total = 0
            target.copy_from(model)

        # Get new experience
        action = model.sample_action(current_state, epsilon=epsilon)

        # update the current state
        prev_state = current_state
        obs, reward, done, _ = env.step(action)
        current_state = update_state(current_state, obs)

        # add new experience sample
        exp = (prev_state, action, reward, current_state, done)
        model.add_experience(exp)
        model.update(target)

        t_total += 1
        total_ep_reward += reward

    return total_ep_reward, t_total


if __name__ == "__main__":

    env = gym.make('Breakout-v0')
    current_state = env.reset()

    # print(type(current_state))
    # print(current_state.shape)
    #
    # figure()
    # imshow(current_state)
    # show()

    # Create agent
    model = DQN(
        state_shape=(80, 80, 4),
        num_actions=env.action_space.n,
        gamma=0.99,
        learning_rate=0.001,
        exp_buf_size=500000,
        min_train_exp=50000,
        batch_size=32,
        scope="model")

    # Create target agent
    target = DQN(
        state_shape=(80, 80, 4),
        num_actions=env.action_space.n,
        gamma=0.99,
        learning_rate=0.001,
        exp_buf_size=500000,
        min_train_exp=50000,
        batch_size=32,
        scope="target")

    with tf.Session() as sess:
        model.set_session(sess)
        target.set_session(sess)
        sess.run(tf.global_variables_initializer())

        # decays linearly until 0.1
        epsilon = 1.0
        epsilon_min = 0.1
        epsilon_change = (epsilon - epsilon_min) / 500000

        num_episodes = 10000
        t_total = 0
        update_t = 10000
        ep_rewards = np.zeros(num_episodes)

        for n in range(num_episodes):

            total_ep_reward, t_total = play_episode(env, model, target, epsilon, t_total, update_t)
            ep_rewards[n] = total_ep_reward

        # get the running average of the rewards
        running_avg = np.empty(num_episodes)
        for t in range(num_episodes):
            running_avg[t] = ep_rewards[max(0, t - 100):(t + 1)].mean()

        figure(figsize=(16, 9))
        plot(ep_rewards)
        plot(running_avg)
        title("Rewards")
        show()

        # # Monitoring visually
        # filename = os.path.basename(__file__).split('.')[0]
        # monitor_dir = './' + filename + '_' + str(datetime.datetime.now())
        # env = wrappers.Monitor(env, monitor_dir)
        #
        # cost, totalreward = play_episode(env, model, tmodel, epsilon=0.0, copy_period=copy_period)