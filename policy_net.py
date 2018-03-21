from __future__ import print_function
import os
import tensorflow as tf
from state import State
from util import Log

# conv
class PolicyNetConv:
    def __init__(self):
        self.board_size = State.BOARD_SIZE
        self.input_channels = 3
        self.init_network()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=100)

    def save_network(self, global_step):
        save_path = os.path.join('models', 'policy_net')
        save_path = self.saver.save(self.session, save_path, global_step=global_step)
        Log.log('saved in:', save_path)

    def load_network(self, model_name):
        self.saver.restore(self.session, model_name)
        Log.log('restored:', model_name)


    @staticmethod
    def _get_weight(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    @staticmethod
    def _get_bias(shape):
        return tf.Variable(tf.constant(0.1, shape=shape))

    def init_network(self):
        self.channels_conv1 = 30
        self.channels_conv2 = 60
        self.channels_conv3 = 90

        # (batch, height, width, channels)
        self.state_input = tf.placeholder(tf.float32, shape=(None, self.board_size, self.board_size, self.input_channels))
        # self.conv1 = tf.nn.conv2d(
        #     input=self.state_input,
        #     filter=self._get_weight([3, 3, 2, self.channels_conv1]),
        #     padding='SAME',
        #     strides=[1, 1, 1, 1]
        # ) + self._get_bias([self.channels_conv1])   # (None, 9, 9, 30)
        # self.relu1 = tf.nn.relu(self.conv1)
        #
        # self.conv2 = tf.nn.conv2d(
        #     input=self.relu1,
        #     filter=self._get_weight([3, 3, self.channels_conv1, self.channels_conv2]),
        #     padding='SAME',
        #     strides=[1, 1, 1, 1]
        # ) + self._get_bias([self.channels_conv2])
        # self.relu2 = tf.nn.relu(self.conv2)
        #
        # self.conv3 = tf.nn.conv2d(
        #     input=self.relu2,
        #     filter=self._get_weight([3, 3, self.channels_conv2, self.channels_conv3]),
        #     padding='SAME',
        #     strides=[1, 1, 1, 1]
        # ) + self._get_bias([self.channels_conv3])
        # self.relu3 = tf.nn.relu(self.conv3)
        # relu3_size = self.relu3.shape[1] * self.relu3.shape[2] * self.relu3.shape[3]
        #
        # self.relu3_vec = tf.reshape(self.relu3, shape=(-1, int(relu3_size)))
        # self.action_prob = tf.layers.dense(inputs=self.relu3_vec, units=self.board_size * self.board_size, activation=tf.nn.softmax)
        # self.value = tf.layers.dense(inputs=self.relu3_vec, units=1, activation=tf.nn.tanh)

        self.conv1 = tf.layers.conv2d(inputs=self.state_input, filters=30, kernel_size=3, strides=1, padding='SAME',
                                      activation=tf.nn.relu, name='conv1')
        self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=60, kernel_size=3, strides=1, padding='SAME',
                                      activation=tf.nn.relu, name='conv2')
        self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=120, kernel_size=3, strides=1, padding='SAME',
                                      activation=tf.nn.relu, name='conv3')

        # action probs
        self.conv_policy = tf.layers.conv2d(self.conv3, filters=4, kernel_size=1, strides=1, padding='SAME',
                                            activation=tf.nn.relu)  # (?, 9, 9, 4)
        self.conv_policy_vec = tf.reshape(self.conv_policy, shape=(-1, int(self.conv_policy.shape[1] * self.conv_policy.shape[2] * self.conv_policy.shape[3])))
        action_prob = tf.layers.dense(inputs=self.conv_policy_vec, units=self.board_size * self.board_size)
        self.action_prob = tf.nn.softmax(action_prob)

        # value
        self.conv_value = tf.layers.conv2d(self.conv3, filters=2, kernel_size=1, strides=1, padding='SAME',
                                           activation=tf.nn.relu)
        self.conv_value_vec = tf.reshape(self.conv_value, shape=(-1, int(self.conv_value.shape[1] * self.conv_value.shape[2] * self.conv_value.shape[3])))
        # print(self.conv_value_vec.shape)
        self.value_dense = tf.layers.dense(self.conv_value_vec, units=64, activation=tf.nn.relu)
        self.value = tf.layers.dense(self.value_dense, units=1, activation=tf.nn.tanh)

        # ground truth
        self.action_prob_input = tf.placeholder(tf.float32, shape=self.action_prob.shape)
        self.value_input = tf.placeholder(tf.float32, shape=self.value.shape)

        # training loss
        # self.action_prob_loss = -tf.reduce_sum(self.action_prob_input * tf.log(self.action_prob))   # cross entropy
        self.l2_penal = 0
        for v in tf.trainable_variables():
            if not 'bias' in v.name.lower():
                self.l2_penal += tf.nn.l2_loss(v)

        # self.action_prob_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.action_prob, labels=self.action_prob_input))
        # cross entropy
        self.action_prob_loss = tf.reduce_mean(-tf.reduce_sum(self.action_prob_input * tf.log(self.action_prob), axis=1))
        # cross entropy
        # self.action_prob_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=action_prob, labels=self.action_prob_input))
        self.value_loss = tf.losses.mean_squared_error(self.value_input, self.value)
        self.loss = (self.action_prob_loss + self.value_loss) + (self.l2_penal * 1e-4)

        self.train_step = tf.placeholder(tf.int32)
        # self.learning_rate = tf.train.exponential_decay(learning_rate=1e-3, global_step=self.train_step, decay_steps=10, decay_rate=0.9)
        # self.learning_rate = tf.train.exponential_decay(learning_rate=1e-3, global_step=self.train_step, decay_steps=30,
        #                                                 decay_rate=0.95)
        self.learning_rate = tf.placeholder(tf.float32)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        # self.train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.loss)


def main():
    net = PolicyNetConv()

if __name__ == '__main__':
    main()