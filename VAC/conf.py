

MAX_EPISODES = 200                          # 最大迭代次数
MAX_EP_STEPS = 46                           # 最大步数
LR_A = 1e-4                                 # actor的学习率
LR_C = 1e-4                                 # critic的学习率
GAMMA = 0.9                                 # 奖励值衰减
REPLACE_ITER_A = 800                        # actor目标网络参数的更新周期
REPLACE_ITER_C = 700                        # critic目标网络参数的更新周期
MEMORY_CAPACITY = 32                        # 记忆库的容量
BATCH_SIZE = 8                              # batch大小
VAR_MIN = 0.1                               # 探索率的衰减
LOAD = False                                # 是否load强化学习模型
