
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior() 

import numpy as np

from nba_env import NbaEnv

np.random.seed(1)
tf.set_random_seed(1)
#tf.random.set_seed(1)

env = NbaEnv()
STATE_DIM = env.state_dim
ACTION_DIM = env.action_dim
ACTION_BOUND = env.action_bound

# all placeholder for tf
with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s_')


class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, t_replace_iter):
        self.sess = sess
        self.a_dim = action_dim                                                                     # 动作的维度
        self.action_bound = action_bound                                                            # 动作的范围
        self.lr = learning_rate                                                                     # actor网络的学习率
        self.t_replace_iter = t_replace_iter                                                        # 目标网络参数更新的周期
        self.t_replace_counter = 0                                                                  # 记录迭代次数，和上一个变量配合更新目标网络

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(S, scope='eval_net', trainable=True)                           # 创建actor的估计网络，参数实时更新

            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)                      # 创建actor的目标网络，参数定期从估计网络拷贝过来

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')    # 估计网络的参数
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')  # 目标网络的参数

    def _build_net(self, s, scope, trainable):
        """
        搭建Actor网络
        :param s: 状态
        :param scope:命名空间
        :param trainable: 参数是否可被训练
        :return: 动作
        """
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.001)
            net = tf.layers.dense(s, 100, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)
            net = tf.layers.dense(net, 20, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          name='a', trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound,
                                       name='scaled_a')
        return scaled_a

    def learn(self, s):
        """
        批训练，更新actor估计网路的参数
        每t_replace_iter次迭代，将估计网路参数直接赋值给目标网络
        :param s: 状态
        :return:
        """
        self.sess.run(self.train_op, feed_dict={S: s})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1

    def choose_action(self, s):
        """
        将状态输入到actor目标网络中得到动作
        :param s: agent的状态
        :return: agent的动作
        """
        s = s[np.newaxis, :]
        return self.sess.run(self.a, feed_dict={S: s})[0]

    def add_grad_to_graph(self, a_grads):
        """
        使用RMSProp算法对actor网络求最优化
        :param a_grads:
        :return:
        """
        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.RMSPropOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, t_replace_iter, a, a_):
        self.sess = sess
        self.s_dim = state_dim                                                                          # 状态的维度
        self.a_dim = action_dim                                                                         # 动作的维度
        self.lr = learning_rate                                                                         # critic网络的学习率
        self.gamma = gamma                                                                              # 奖励的衰减值，用于贝尔曼方程计算td_error
        self.t_replace_iter = t_replace_iter                                                            # 目标网络参数更新的周期
        self.t_replace_counter = 0

        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = a
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)                             # 创建critic估计网络

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, 'target_net',                                             # 创建critic目标网络
                                      trainable=False)                                                  # 目标q值的计算需要actor的目标网络输出作为输入

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')   # 估计网络的参数
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net') # 目标网络的参数

        with tf.variable_scope('target_q'):                                                             # 通过贝尔曼方程计算目标q值
            self.target_q = R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):                                                             # 目标q和估计网络输出的q值的均方误差作为td_error，即损失函数
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):                                                              # 采用RMSProp算法最小化loss
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):                                                               # 每个样本的梯度张量
            self.a_grads = tf.gradients(self.q, a)[0]

    def _build_net(self, s, a, scope, trainable):
        """
        搭建critic网络
        :param s: 状态
        :param a: 动作
        :param scope:参数的命名空间
        :param trainable: 是否可被训练
        :return: q值
        """
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.01)

            with tf.variable_scope('l1'):
                n_l1 = 100
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, 20, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b,
                                    trainable=trainable)  # Q(s,a)
        return q

    def learn(self, s, a, r, s_):
        """
        批训练，更新critic估计网路的参数
        每t_replace_iter次迭代，将估计网路参数直接赋值给目标网络
        :param s: 状态
        :return:
        """
        self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r, S_: s_})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1


class Memory(object):
    def __init__(self, capacity, dims):
        """
        初始化记忆库
        :param capacity: 记忆库的容量
        :param dims:记忆库的维度
        """
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0                                                        # 指向记忆库的当前位置

    def store_transition(self, s, a, r, s_):
        """
        将s,a,r,s_保存在记忆库中，若记忆库满，则从头覆盖
        :param s: 当前状态
        :param a: 动作
        :param r: 在当前状态s做动作a，环境反馈回的奖励
        :param s_: 在当前状态s做动作a，环境反馈回的下一个状态
        :return:
        """
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        """
        从记忆库中随机挑选n条记录
        :param n: 挑选的记录个数
        :return: 挑选的记录
        """
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]

