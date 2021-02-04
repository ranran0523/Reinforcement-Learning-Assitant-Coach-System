
import numpy as np
import pandas as pd
from keras.models import load_model


class NbaEnv:
    action_dim = 8                                                  # 动作的维度
    state_dim = 42                                                  # 状态的维度

    def __init__(self):
        """
        从csv文件中读取状态和dal
        """
        state_data = pd.read_csv("./Dalteamdata.csv").to_numpy()
        self.state_data = np.delete(state_data, 0, axis=1)          # 删掉第一列

        dal_data = pd.read_csv("./DAL.csv").to_numpy()
        self.dal_data = np.delete(dal_data, [0, 1, 2, 3, 4, 5, 6, 7, 8], axis=1)    # 删掉前9列

        self.action_bound = [0, 48]                                 # 每个actor在0到48之间

        self.step_num = 1                                           # 记录迭代到哪一状态

        # TODO:模型1和模型2的加载问题
        # self.model1 = load_model('my_model_1.h5')
        # self.model1.summary()
        # self.model2 = load_model('my_model_2.h5')
        # self.model2.summary()

    def step(self, action):
        """
        计算reward并返回状态和reward
        :param action:
        :return: 第三维：判断是否结束迭代
        """
        model1_input = np.hstack((action, self.dal_data[self.step_num]))
        # model1_output = self.model1.predict(model1_input)
        model1_output = np.random.random(42)                                        # TODO:模型1和2加载成功后删掉这行，打开上一行注释
        model2_input = np.hstack((model1_output, self.state_data[self.step_num]))
        # model2_output = self.model2.predict(model2_input)
        model2_output = np.random.random()                                          # TODO:模型1和2加载成功后删掉这行，打开上一行注释
        if model2_output > 0.5:
            reward = 1
        else:
            reward = 0
        
        win= reward
        fatigue = 1 - sum(action) / 240
        reward = reward + fatigue * 0.5
        self.step_num += 1
        return self.state_data[self.step_num - 1], reward, win ,self.step_num > 45

    def reset(self):
        """
        初始化状态
        :return: 起始状态
        """
        self.step_num = 0
        return self.state_data[self.step_num]

    def sample_action(self):
        a = np.random.uniform(*self.action_bound, size=self.action_dim)
        return a


if __name__ == '__main__':
    env = NbaEnv()
    for ep in range(1):
        s = env.reset()
        print(s)
        while True:
            s, r, done = env.step(env.sample_action())
            print(s)
            if done:
                break

