import numpy as np
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Lambda, Activation, Dropout, Concatenate
from keras.layers import Dense, Embedding, LSTM, Conv1D, GlobalMaxPooling1D
from keras.layers.wrappers import Bidirectional
from keras.layers import Activation
from keras.layers.wrappers import TimeDistributed
from keras.utils import to_categorical
import tensorflow as tf
import pickle
import linecache
import cx_Oracle

def train_rules(self, rule_len, path):
    print("执行了train_rules")
    rules_idx = rule_len
    print("序列样本数量为", rules_idx)
    f = open('data/save/id2word.txt', 'r')
    id2word = eval(f.read())
    f.close()
    f = open('data/save/word2id.txt', 'r')
    word2id = eval(f.read())
    f.close()
    f = open('data/save/rules_final.txt', 'r')
    try:
        rules_final = {}  # 若想每次在上一版的基础上改进，则为eval(f.read())
    except:
        rules_final = {}
    f.close()

    for idx in range(rules_idx):
        sentence = linecache.getline(path, idx)  # 读取原始数据中的第idx行
        words = sentence.strip().split(",")  # list类型
        # print()
        # print("sentence",sentence)          #增加可解释性时显示
        # print("words:",words)               #words: ['31301', 'BENSON', 'AZ', '85602', 'COCHISE']
        # print("words[0]:",words[0])
        f = open('data/save/att_name.txt', 'r')
        label2att = eval(f.read())
        f.close()

        # LHS多属性函数依赖，此处若想恢复在生成规则时就进行筛选，查看E盘备份的此模块
        reason = words  # words: ['31301', 'BENSON', 'AZ', '85602', 'COCHISE']
        self.reset_rnn_state()  # 重置LSTM状态
        action = np.zeros([self.B, 1], dtype=np.int32)
        # print("self.B",self.B)
        for i in range(len(reason) - 2):  # len(reason)
            # print("___________________")
            flag = i  # i是当前reason部分的最后一位，flag是标记，是当然reason中第一个元素所在位置，当flag<0，代表前面已经没用元素，即reason部分无法推出result，放弃该规则
            flag_ = flag  # 新增信息应对的索引，从reason的最后一位开始向前移动

            while (flag >= 0):
                # print("flag",flag)
                sqlex = ""
                dic_name = []
                left_ = []  # 用以存贮构建字典的原因部分
                word_ = []  # 用以存贮构建字典的名称,其实这个与dic_name相同
                while (flag_ != i):
                    # print("i=",i,"flag=",flag,"flag_=",flag_)
                    word = reason[flag_]
                    left = label2att[flag_]  # 新增信息
                    try:
                        action[0][0] = word2id[word]
                    except:
                        print("字典中无", word)
                        action[0][0] = '3'

                    # print("增加信息",action,"即",left,":",word)    #增加可解释性时显示
                    prob = self.predict(action)
                    dic_name.append(word)
                    left_.append(left)
                    word_.append(word)
                    flag_ = flag_ + 1
                word = reason[flag_]  # word: 31301;word: BENSON;word: AZ;word: 85602;word: COCHISE，每次一个
                dic_name.append(word)
                # print(word)
                left = label2att[flag_]  # 获取字典里对应位置的属性名
                right = label2att[i + 1]

                try:
                    action[0][0] = word2id[word]
                except:
                    action[0][0] = '3'
                    print("字典中无该词")




    print("规则生成完成，数量为", len(rules_final))
    # print(str(rules_final))