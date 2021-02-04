import tensorflow as tf
import os
import shutil
from DDPG import *
from nba_env import *
from conf import *

sess = tf.Session()

# 创建actor和critic网络
actor = Actor(sess, ACTION_DIM, ACTION_BOUND[1], LR_A, REPLACE_ITER_A)
critic = Critic(sess, STATE_DIM, ACTION_DIM, LR_C, GAMMA, REPLACE_ITER_C, actor.a, actor.a_)
actor.add_grad_to_graph(critic.a_grads)

# 创建记忆库
M = Memory(MEMORY_CAPACITY, dims=2 * STATE_DIM + ACTION_DIM + 1)

saver = tf.train.Saver()
path = './save'

# 如果LOAD为True则加载模型，否则初始化参数
if LOAD:
    saver.restore(sess, tf.train.latest_checkpoint(path))
else:
    sess.run(tf.global_variables_initializer())


def train():
    var = 2.                                                        # 探索度
    reward = 0                                                      # 记录每次完整step的reward
    for ep in range(MAX_EPISODES):
        s = env.reset()                                             # 环境初始化，返回初始状态
        ep_step = 0

        for t in range(MAX_EP_STEPS):

            # Added exploration noise
            a = actor.choose_action(s)                              # 通过actor目标网络得到当前动作
            a = np.clip(np.random.normal(a, var), *ACTION_BOUND)    # 给动作加上用于探索的随机量，并截断到0-48
            s_, r, done = env.step(a)                               # 将动作作用到环境中，得到下一个状态和环境反馈的奖励，是否结果迭代
            M.store_transition(s, a, r, s_)                         # 将s, a, r, s_保存到记忆库中
            reward += r                                             # 加上本次step的奖励

            if M.pointer > MEMORY_CAPACITY:                         # 当记忆库的记录容量达到MEMORY_CAPACITY，开始训练
                var = max([var * .9995, VAR_MIN])                   # 对动作的随机探索率进行衰减
                b_M = M.sample(BATCH_SIZE)                          # 随机从记忆库中取出一个BATCH的数据进行训练
                b_s = b_M[:, :STATE_DIM]
                b_a = b_M[:, STATE_DIM: STATE_DIM + ACTION_DIM]
                b_r = b_M[:, -STATE_DIM - 1: -STATE_DIM]
                b_s_ = b_M[:, -STATE_DIM:]

                critic.learn(b_s, b_a, b_r, b_s_)                   # 更新critic估计网络的参数
                actor.learn(b_s)                                    # 更新actor估计网络的参数

            s = s_                                                  # 将下一个状态作为下一次迭代的当前状态
            ep_step += 1

            if done or t == MAX_EP_STEPS - 1:                       # 当完整step结束后，即46场比赛结束后，打印信息
                print('Ep:', ep,
                      '| Steps: %i' % int(ep_step),
                      '| Explore: %.2f' % var,
                      '| Reward: %.2f' % reward
                      )
                reward = 0
                break

    if os.path.isdir(path): shutil.rmtree(path)                     # 保存参数
    os.mkdir(path)
    ckpt_path = os.path.join(path, 'DDPG.ckpt')
    save_path = saver.save(sess, ckpt_path, write_meta_graph=False)
    print("\nSave Model %s\n" % save_path)


def eval():
    while True:
        s = env.reset()
        while True:
            a = actor.choose_action(s)
            s_, r, done = env.step(a)
            s = s_
            if done:
                break


if __name__ == '__main__':
    if LOAD:
        eval()
    else:
        train()