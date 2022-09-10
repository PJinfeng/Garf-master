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
import math
import json
from SeqGAN.utils import Vocab,load_data

import code

def GeneratorPretraining(V, E, H):
    '''
    Model for Generator pretraining. This model's weights should be shared with
        Generator.
    # Arguments:
        V: int, Vocabrary size
        E: int, Embedding size
        H: int, LSTM hidden size
    # Returns:
        generator_pretraining: keras Model
            input: word ids, shape = (B, T)
            output: word probability, shape = (B, T, V)
    '''
    # in comment, B means batch size, T means lengths of time steps.
    input = Input(shape=(None,), dtype='int32', name='Input') # (B, T)
    out = Embedding(V, E, mask_zero=True, name='Embedding')(input) # (B, T, E)      #构建Layer之间的函数链式关系,Embedding层用以改变输出形状
    out = LSTM(H, return_sequences=True, name='LSTM')(out)  # (B, T, H)             #构建Layer之间的函数链式关系
    out = TimeDistributed(                                               #TimeDistributed层对每一个向量进行了一个Dense操作
        Dense(V, activation='softmax', name='DenseSoftmax'),             #定义一个有V个节点，使用softmax激活函数的神经层
        name='TimeDenseSoftmax')(out)    # (B, T, V)
    generator_pretraining = Model(input, out)
    return generator_pretraining

class Generator():
    'Create Generator, which generate a next word.'

    # def rule_dict(self):


    def __init__(self, sess, B, V, E, H, lr=1e-3):
        '''
        # Arguments:
            B: int, Batch size
            V: int, Vocabrary size
            E: int, Embedding size
            H: int, LSTM hidden size
        # Optional Arguments:
            lr: float, learning rate, default is 0.001
        '''
        self.sess = sess
        self.B = B
        self.V = V
        self.E = E
        self.H = H
        self.lr = lr
        self._build_gragh()
        self.reset_rnn_state()


    def _build_gragh(self):
        state_in = tf.placeholder(tf.float32, shape=(None, 1))      #传入数据，其中None指的batch size的大小，可以是任何数，1是指的数据的尺寸
        h_in = tf.placeholder(tf.float32, shape=(None, self.H))     #(B,H)
        c_in = tf.placeholder(tf.float32, shape=(None, self.H))     #(B,H)
        action = tf.placeholder(tf.float32, shape=(None, self.V))   #onehot (B, V)
        reward  =tf.placeholder(tf.float32, shape=(None, ))         #(B, )每一个batch的reward

        self.layers = []

        embedding = Embedding(self.V, self.E, mask_zero=True, name='Embedding')     #第一层为Embedding层
        out = embedding(state_in)                                                   #输入为(B,V)，输出(B,1,E)
        self.layers.append(embedding)

        lstm = LSTM(self.H, return_state=True, name='LSTM')                         #lstm层
        # out, next_h, next_c = Bidirectional(lstm(out, initial_state=[h_in, c_in]))                 #输入为（B,1,E）和2个（B,H）,输出为 (B, H)
        out, next_h, next_c = lstm(out, initial_state=[h_in, c_in])                 #输入为（B,1,E）和2个（B,H）,输出为 (B, H)
        self.layers.append(lstm)

        dense = Dense(self.V, activation='softmax', name='DenseSoftmax')            #全连接层
        prob = dense(out)                                                           #输入为（B,H），输出为（B,V）
        self.layers.append(dense)

        log_prob = tf.log(tf.reduce_mean(prob * action, axis=-1)) # (B, )         取每一行数据与onehot的action做乘法后，取平均值的对数
        loss = - log_prob * reward
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        minimize = optimizer.minimize(loss)
        #以下操作为整体训练lstm，而本文需要单步进行，每次更新由rl指导，因此不进行下列操作
        #model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])  #编译模型
        #print(model.summary())                                                                   #显示模型结构
        #model.fit(data, labels)  # starts training                                               #拟合网络
        #loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)                         #测试
        #classes = model.predict_classes(X_test, batch_size=32)
        #proba = model.predict_proba(X_test, batch_size=32)                                       #使用

        self.state_in = state_in
        self.h_in = h_in
        self.c_in = c_in
        self.action = action
        self.reward = reward
        self.prob = prob
        self.next_h = next_h
        self.next_c = next_c
        self.minimize = minimize
        self.loss = loss

        self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op)

    def reset_rnn_state(self):
        self.h = np.zeros([self.B, self.H])
        self.c = np.zeros([self.B, self.H])

    def set_rnn_state(self, h, c):              #h、c: np.array, shape = (B,H)，ex.（32,64）

        self.h = h
        self.c = c

    def get_rnn_state(self):
        return self.h, self.c

    def predict(self, state, stateful=True):
        '''
        Predict next action(word) probability
        # Arguments:
            state: np.array, previous word ids, shape = (B, 1)
        # Optional Arguments:
            stateful: bool, default is True
                if True, update rnn_state(h, c) to Generator.h, Generator.c
                    and return prob.
                else, return prob, next_h, next_c without updating states.
        # Returns:
            prob: np.array, shape=(B, V)
        '''
        # state = state.reshape(-1, 1)
        feed_dict = {
            self.state_in : state,
            self.h_in : self.h,
            self.c_in : self.c}
        prob, next_h, next_c = self.sess.run(               #prob：np.array，shape=（B,V）ex.（32,1398）
            [self.prob, self.next_h, self.next_c],
            feed_dict)
        # print(prob.shape)
        # print(next_c.shape)
        # print(next_c)

        if stateful:
            self.h = next_h
            self.c = next_c
            return prob
        else:
            return prob, next_h, next_c

    def update(self, state, action, reward, h=None, c=None, stateful=True):

        if h is None:
            h = self.h
        if c is None:
            c = self.c
        state = state[:, -1].reshape(-1, 1)
        reward = reward.reshape(-1)
        feed_dict = {
            self.state_in : state,
            self.h_in : h,
            self.c_in : c,
            self.action : to_categorical(action, self.V),
            self.reward : reward}
        _, loss, next_h, next_c = self.sess.run(
            [self.minimize, self.loss, self.next_h, self.next_c],
            feed_dict)

        if stateful:
            self.h = next_h
            self.c = next_c
            return loss
        else:
            return loss, next_h, next_c

    def sampling_word(self, prob):

        action = np.zeros((self.B,), dtype=np.int32)
        for i in range(self.B):
            p = prob[i]                                        #p是一个1维V列的数组
            # print("p:",p)
            p /=p.sum()                                        #总概率归一化
            action[i] = np.random.choice(self.V, p=p)          #根据p提供的概率，概率地选择从0到V之间的一个数
        #     print("action[i]",action[i])
        # print("action",action)
        return action

    def sampling_sentence(self, T, BOS=1):                     #根据rnn生成句子

        self.reset_rnn_state()                                  #参数h,c状态重置
        action = np.zeros([self.B, 1], dtype=np.int32)          #生成大小为B的纵向数组，初始全0
        action[:, 0] = BOS                                      #首位全置BOS
        actions = action
        # print(T)
        for _ in range(T):                                      #T是一句话的最大长度
            prob = self.predict(action)                         #根据当前序列预测后续序列，返回参数
            # print("prob:",prob)
            action = self.sampling_word(prob).reshape(-1, 1)    #根据参数采样单词，并转成1列
            # print("action:",action)
            # print(len(action))
            for i in range(len(action)):
                if (action[i][0] == 4):
                    action[i][0] = 3
                    # print("将none转化为unknown")
            # print("转变后action:",action)
            actions = np.concatenate([actions, action], axis=-1)#由word的id组成的B个句子的合集
            # print("actions:",actions)
        # Remove BOS
        actions = actions[:, 1:]        #取所有数据的第1列到右侧数据，即除第0列的右侧所有
        # print(actions)
        self.reset_rnn_state()

        return actions

    def generate_samples(self, T, g_data, num, output_file):
        print("执行了generate_samples")
        sentences=[]

        for _ in range(num // self.B + 1):
            actions = self.sampling_sentence(T)                 #根据神经网络，生成句子，返回的是word的id所构成的array

            actions_list = actions.tolist()                     #从array变为list

            for sentence_id in actions_list:
                # print("sentence_id",sentence_id)
                sentence = [g_data.id2word[action] for action in sentence_id if action != 0 and action != 2]     #将id反转成word, 此句是上面的简洁写法
                sentences.append(sentence)                      #生成器生成的句子
                # print(sentences)

        output_str = ''

        for i in range(num):
            # print(sentences[i])
            # if (sentences[i] is None):
            #     sentences[i]=""
            output_str += ' '.join(sentences[i]) + '\n'
            # print(output_str)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_str)
        print("生成的序列已写入",output_file)

    def sampling_rule(self,   T, BOS=1):                     #根据rnn生成规则
        # print("执行了sampling_rule")

        f = open('data/save/id2word.txt', 'r')
        id2word = eval(f.read())
        f.close()
        # print(id2word[684])
        f = open('data/save/word2id.txt', 'r')
        word2id = eval(f.read())
        # print(word2id)
        f.close()

        self.reset_rnn_state()
        action = np.zeros([self.B, 1], dtype=np.int32)          #生成大小为B的纵向数组，初始全0
        actions = np.empty([self.B, 0], dtype=np.int32)
        # print("actions",actions)
        action[:, 0] = BOS                                      #首位全置BOS
        for _ in range(T):                                      #T是一句话的最大长度

            prob = self.predict(action)                         #根据当前序列预测后续序列，返回参数，输入到网络中的形状为（B,1）

            action = self.sampling_word(prob).reshape(-1, 1)    #根据参数采样单词，并转成1列
            # print("训练时预测输出:", action,id2word[action[0][0]])
            action_=np.argmax(prob, axis=-1).reshape([-1, 1])
            # print("训练时预测输出_:", action_,id2word[action_[0][0]])
            # print("_________________")
            show2 = []
            for id in action:
                # print(id)
                word = id2word[id[0]]
                show2.append(word)
            # print("预测选择:",show2)

            show3 = np.array(show2).reshape(-1, 1)
            # print(show3)
            actions = np.concatenate([actions, show3], axis=-1) #规则生长
            # print("当前规则:",actions)
        self.reset_rnn_state()               #重置lstm的状态，很重要，删掉相当于接着上面的话继续预测，而不是根据当前状态预测


        return actions

    def generate_rules(self, T, g_data, num, output_file):
        print("执行了generate_rules")
        # print(output_file)
        rules=[]
        for _ in range(num // self.B + 1):
            actions = self.sampling_rule(T)                 #根据神经网络，生成句子
            # print(actions.shape)                            #array,大小为(B,T)
            print(actions)
            actions_list = actions.tolist()                 #从array变为list,维度不变, ex.(32,7)

            for rule_word in actions_list:                    #循环B次
                rule = rule_word
                # print(rule)
                rules.append(rule)                      #生成器生成的句子
                # print(rules)

        output_str = ''

        for i in range(num):
            # print(rules[i])
            for n in range(len(rules[i])):
                if (rules[i][n] == None):
                    rules[i][n] = '<UNK>'
            # print(rules[i])
            output_str += ','.join(rules[i]) + '\n'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_str)
        print("生成样本数量为",num)
        print("已写入", output_file)

    def predict_rules(self):                    #生成单对单的规则
        print("执行了predict_rules")
        f = open('data/save/id2word.txt', 'r')
        id2word = eval(f.read())
        f.close()
        f = open('data/save/word2id.txt', 'r')
        word2id = eval(f.read())
        f.close()

        self.reset_rnn_state()
        reason = np.zeros([self.B, 1], dtype=np.int32)  # 生成大小为B的纵向数组，初始全0
        word = '10001'#10001,DOTHAN,KINGMAN,POCAHONTAS,21303;BOAZ是错的
        reason[0][0] = word2id[word]
        # print("输入的原因部分:", reason, id2word[reason[0][0]])  # ,action.shape,type(action)
        prob = self.predict(reason)
        result = np.argmax(prob, axis=-1).reshape([-1, 1])
        # print("预测输出最大可能性的结果:", result, id2word[result[0][0]])
        result_ = np.random.choice(self.V, p=prob[0])
        # print("随概率分布进行预测输出_:", result_, id2word[result_])



    def multipredict_rules_argmax(self,reason):    #生成多对单的规则
        print("执行了multipredict_rules")
        f = open('data/save/id2word.txt', 'r')
        id2word = eval(f.read())
        f.close()
        f = open('data/save/word2id.txt', 'r')
        word2id = eval(f.read())
        f.close()

        self.reset_rnn_state()
        action = np.zeros([self.B, 1], dtype=np.int32)
        # print(action.shape)
        # print(action)
        # reason=['10005','BOAZ','AL','36251']#['10005','BOAZ']
        # print(type(reason))
        for i in range(len(reason)):
            word = reason[i]
            action[0][0] = word2id[word]
            prob = self.predict(action)
            result = np.argmax(prob, axis=-1).reshape([-1, 1])
            result = id2word[result[0][0]]
            #暂时注释掉下面
            #result_ = np.random.choice(self.V, p=prob[0])
            #result_ = id2word[result_]
            #这部分显示可恢复
            # if (i==len(reason)-1):
            #     print("输入的原因部分:", reason)  # ,action.shape,type(action)
            #     print("预测输出最大可能性的结果:", result, )
            #     print("随概率分布进行预测输出_:", result_, )
        return result

    def multipredict_rules_probability(self,reason):    #生成多对单的规则
        print("执行了multipredict_rules")
        f = open('data/save/id2word.txt', 'r')
        id2word = eval(f.read())
        f.close()
        f = open('data/save/word2id.txt', 'r')
        word2id = eval(f.read())
        f.close()

        self.reset_rnn_state()
        action = np.zeros([self.B, 1], dtype=np.int32)
        # print(action.shape)
        # print(action)
        # reason=['10005','BOAZ','AL','36251']#['10005','BOAZ']
        # print(type(reason))
        for i in range(len(reason)):
            word = reason[i]
            try:
                action[0][0] = word2id[word]
            except:
                action[0][0] = '3'

            prob = self.predict(action)
            result = np.argmax(prob, axis=-1).reshape([-1, 1])
            result = id2word[result[0][0]]
            result_ = np.random.choice(self.V, p=prob[0])
            result_ = id2word[result_]
            #这部分显示可恢复
            # if (i==len(reason)-1):
            #     print("输入的原因部分:", reason)  # ,action.shape,type(action)
            #     print("预测输出最大可能性的结果:", result, )
            #     print("随概率分布进行预测输出_:", result_, )
        return result_

    def train_rules(self,rule_len,path):
        print("执行了train_rules")
        rules_idx = rule_len
        print("序列样本数量为",rules_idx)
        f = open('data/save/id2word.txt', 'r')
        id2word = eval(f.read())
        f.close()
        f = open('data/save/word2id.txt', 'r')
        word2id = eval(f.read())
        f.close()
        f = open('data/save/rules_final.txt', 'r')
        try:
            rules_final = {}#若想每次在上一版的基础上改进，则为eval(f.read())
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
            reason = words                                 #words: ['31301', 'BENSON', 'AZ', '85602', 'COCHISE']
            self.reset_rnn_state()                         #重置LSTM状态
            action = np.zeros([self.B, 1], dtype=np.int32)
            # print("self.B",self.B)
            for i in range(len(reason)-2):  # len(reason)
                # print("___________________")
                flag=i      #i是当前reason部分的最后一位，flag是标记，是当然reason中第一个元素所在位置，当flag<0，代表前面已经没用元素，即reason部分无法推出result，放弃该规则
                flag_=flag  #新增信息应对的索引，从reason的最后一位开始向前移动

                while(flag>=0):
                    # print("flag",flag)
                    sqlex = ""
                    dic_name=[]
                    left_=[]        #用以存贮构建字典的原因部分
                    word_=[]        #用以存贮构建字典的名称,其实这个与dic_name相同
                    while (flag_!=i):
                        # print("i=",i,"flag=",flag,"flag_=",flag_)
                        word = reason[flag_]
                        left = label2att[flag_]  # 新增信息
                        try:
                            action[0][0] = word2id[word]
                        except:
                            print("字典中无",word)
                            action[0][0] = '3'

                        # print("增加信息",action,"即",left,":",word)    #增加可解释性时显示
                        prob = self.predict(action)
                        dic_name.append(word)
                        left_.append(left)
                        word_.append(word)
                        flag_=flag_+1
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

                    prob = self.predict(action)
                    result = np.argmax(prob, axis=-1).reshape([-1, 1])  # 字典索引
                    result = id2word[result[0][0]]  # 实际内容
                    # result_ = np.random.choice(self.V, p=prob[0])
                    # result_ = id2word[result_]
                    # print("输入的原因部分:", word)  # ,action.shape,type(action)
                    # print("预测输出最大可能性的结果:", result)
                    # print("随概率分布进行预测输出_:", result_)
                    # print("___________________")

                    self.reset_rnn_state()


                    #为后续增加部分规则建立字典，但排序需要在前面
                    for n in range(len(left_)):
                        # print("补充字典")
                        # print(rules_final)
                        if (n==0):
                            addtwodimdict(rules_final, str(dic_name), 'reason', {str(left_[n]): str(word_[n])})
                        else:
                            addtwodimdict(rules_final[str(dic_name)], 'reason', str(left_[n]), str(word_[n]))


                    if (i==flag):
                        # print("此时i=flag")
                        addtwodimdict(rules_final,str(dic_name), 'reason',{str(left): str(word)})
                        addtwodimdict(rules_final,str(dic_name), 'result', {str(right): str(result)})
                        # print(rules_final)
                        # rules_final.update({str(dic_name): {'reason': {str(left): str(word)},
                        #                                     'result': {str(right): str(result)}}})
                    else:
                        # print("此时i!=flag")
                        addtwodimdict(rules_final[str(dic_name)], 'reason',str(left), str(word))
                        addtwodimdict(rules_final[str(dic_name)], 'result', str(right), str(result))
                        # print(rules_final)

                    # print("此时原因部分为", rules_final[str(dic_name)]['reason'],"预测值：", rules_final[str(dic_name)]['result'],"实际值为",reason[i+1])    #增加可解释性时显示




                    # 把认为正确的规则保存至字典,删除错误规则
                    if (result==reason[i+1]):
                        # print("预测值与实际值相同，保存规则")       #增加可解释性时显示
                        #推出来了，就跳出循环，否则flag前移一位，加入额外信息继续测试
                        break
                    else:
                        # print("预测值与实际值不符，增加原因部分")    #增加可解释性时显示
                        del rules_final[str(dic_name)]

                    flag=flag-1
                    flag_=flag
                    # if flag<0:
                        # print("已无更多可用信息，reason部分无法推出result，放弃该规则")    #增加可解释性时显示

            f = open('data/save/rules_final.txt', 'w')
            # print(str(rules_final))
            f.write(str(rules_final))
            f.close()

            f = open('data/save/rules_read.txt', 'w')
            # print(str(rules_final))
            for item in rules_final.items():
                f.write(str(item))
                f.write('\r\n')
            f.write(str(rules_final))
            f.close()


        print("规则生成完成，数量为",len(rules_final))
        # print(str(rules_final))

    def filter(self,path):

        print("执行了filter")
        f = open('data/save/att_name.txt', 'r')
        label2att = eval(f.read())
        f.close()
        att2label = {v: k for k, v in label2att.items()}  #字典反向传递
        f = open('data/save/rules_final.txt', 'r')
        rules_final = eval(f.read())
        f.close()
        l1=len(rules_final)
        # print(rules_final)
        num = 0
        conn = cx_Oracle.connect('system', 'Pjfpjf11', '127.0.0.1:1521/orcl')  # 连接数据库
        cursor = conn.cursor()
        for rulename, ruleinfo in list(rules_final.items()):
            num += 1
            # print("过滤第", num, "条规则及对应数据")
            # print("ruleinfo:", ruleinfo)

            left = list(ruleinfo['reason'].keys())
            # print(left)
            word = list(ruleinfo['reason'].values())
            # print(word)
            k = list(ruleinfo['result'].keys())
            right = k[0]
            v = list(ruleinfo['result'].values())
            result = v[0]



            sqlex = left[0] + "\"='" + word[0] + "'"
            i = 1
            while (i < len(left)):
                sqlex = sqlex + " and \"" + left[i] + "\"='" + word[i] + "'"
                i += 1

            sql1 = "select \"" + right + "\" from \"" + path + "\" where \"" + sqlex
            # print(sql1)         #select "MINIT" from "UIS_copy" where "CUID"='9078' and "RUID"='15896' and "SSN"='463210223' and "FNAME"='Monken'
            cursor.execute(sql1)  # "City","State" ,where rownum<=10
            rows = cursor.fetchall()
            num1=len(rows)
            if num1<3:
                # print("满足规则的数据有",num1,"条，推测来源为错误数据，无修复意义，删除规则",rules_final[str(rulename)])
                del rules_final[str(rulename)]
                continue
            else:
                t_rule=1
                for row in rows:
                    if (str(row[-1]) == str(result)):  # 此时规则与数据相符合, 规则置信度增加
                        t_rule = t_rule + 1
                        print("-->", t_rule, end='')
                    else:  # 此时规则与数据相违背, 规则置信度减少
                        t_rule = t_rule - 2
                        print("-->", t_rule, end='')
                        flag = 0  # 标记该规则与数据存在冲突
                rules_final[str(rulename)].update({'confidence': t_rule})  # 规则置信度初始化
                # rules_final[str(rulename)].update({'confidence': 1})
            # sql2 = "select \"" + right + "\" from \"" + path + "\" where \"" + sqlex + " and \"" + right + "\"='" + result + "'"
            # # print(sql2)
            # cursor.execute(sql2)
            # rows = cursor.fetchall()
            # num2 = len(rows)
            # # print(num2)
            # ratio = num2 / num1
            # if ratio < 0.51:
            #     # print("推测来源为错误数据，无修复意义，删除规则")
            #     del rules_final[str(rulename)]
            #     continue

        cursor.close()
        conn.close()


        f = open('data/save/rules_final.txt', 'w')
        f.write(str(rules_final))
        f.close()
        l2=len(rules_final)
        print("规则过滤完成，剩余数量为", )
        print(str(l2))
        # print(str(rules_final))           #过滤后规则
        with open('data/save/log_filter.txt', 'w') as f:
            f.write("原始规则数量为")
            f.write(str(l1))
            f.write("规则过滤后，剩余数量为")
            f.write(str(l2))
            f.write("__________")
        f.close()

    def detect(self,rows,result,rulename,LHS,RHS,att2label,label2att):
        dert = 0
        t0=1
        t_rule=t0
        t_tuple=t0
        t_max=t_tuple   #满足rule条件的不同tuple中置信度最大值
        flag=1         #标记该规则是否与数据存在冲突
        flag_trust = 0  # 0代表相信数据，1代表相信规则
        for row in rows:
            if (str(row[RHS]) == str(result)):
                continue
            else:
                dert += 1
                flag = 0   # 标记该规则与数据存在冲突
        if (flag==1):           #该规则不与数据存在冲突,则直接给一个极大置信度
            t_rule=t_rule+100
            flag_trust = 3  # 3代表规则正确, 且无任何冲突
            return flag_trust
        else:                   #该规则与数据存在冲突,则计算每个tuple的置信度,以调整t_rule
            print("该规则与数据存在冲突")
            print("本次修复预计变化量", dert)
            error_row=[]
            rule_other=[]
            t_rule=t0
            for row in rows:    #每个满足规则条件的tuple
                AV_p=[]
                t_tp = 999   #当前tuple的置信度, 计算为一个tuple中不同AV_i中的置信度最小值,为了避免初始值干扰,先设定个大值
                t_tc = t0
                # flag_p=0     #用以记录AV_p中置信度最小的对应的属性位置
                # rule_p_name=[] #用以记录能够修复上述的AV_p中置信度最小的对应的属性的具有最大置信度的规则
                # print("匹配当前规则的tuple为：", row)
                for i in LHS:       #计算一个tuple中不同AV_i中的置信度最小值
                    AV_p.append(row[i])
                    t_AV_i = t0
                    # rulename_p_max = []
                    # t_rmax = 0
                    attribute_p=label2att[i]
                    for rulename_p, ruleinfo_p in list(self.rule.items()):      #遍历字典
                        if rulename == rulename_p:
                            continue
                        if t_AV_i>100 or t_AV_i<-100:
                            break
                        v = list(ruleinfo_p['result'].values())
                        left = list(ruleinfo_p['reason'].keys())
                        word = list(ruleinfo_p['reason'].values())
                        k = list(ruleinfo_p['result'].keys())
                        t_r = ruleinfo_p['confidence']
                        if t_r<0:
                            continue
                        right = k[0]
                        if attribute_p == right:
                            flag_equal = 0  # 规则能否确定row[i]的标记
                            for k in range(len(left)):
                                if row[att2label[left[k]]] == word[k]:  # 若row[i]所在的tuple满足某条规则的全部AV_p,标记为1
                                    flag_equal = 1
                                else:
                                    flag_equal = 0
                                    break
                            if flag_equal == 1:  # 若该tuple中row[i]能够被其他规则确定,检测其是否满足规则
                                # print(row, "中")
                                # print(right, "可以由其他规则确定：", ruleinfo)
                                result2 = v[0]
                                # if t_rmax < t_r:  # 记录这些规则中最大的规则置信度
                                #     t_rmax = t_rmax
                                #     rulename_p_max = rulename_p  # 记录该最可信规则在字典中的标识
                                if str(row[i]) == str(result2):    # 检索其他规则以确定该tuple中每个Token的置信度,满足则增加,反之则减
                                    t_AV_i = t_AV_i + t_r
                                else:
                                    t_AV_i = t_AV_i - t_r
                                    print("匹配当前规则的tuple为：", row)
                                    print("AV_p中",str(row[i]), "与", str(result2), "不符,对应的规则为", ruleinfo_p, "其置信度为", t_r)

                    if t_tp > t_AV_i:
                        t_tp = t_AV_i
                        # flag_p=i
                        # rule_p_name=rulename_p_max


                for rulename_c, ruleinfo_c in list(self.rule.items()):  # 遍历字典,计算t_c
                    if rulename==rulename_c:
                        continue
                    v = list(ruleinfo_c['result'].values())
                    left = list(ruleinfo_c['reason'].keys())
                    word = list(ruleinfo_c['reason'].values())
                    k = list(ruleinfo_c['result'].keys())
                    t_r = ruleinfo_c['confidence']
                    if t_r < 0:
                        continue
                    right = k[0]
                    attribute_c = label2att[RHS]
                    if attribute_c == right:
                        flag_equal = 0  # 规则能否确定row[i]的标记
                        for k in range(len(left)):
                            if row[att2label[left[k]]] == word[k]:  # 若AV_c所在的tuple满足某条规则的全部AV_p,标记为1
                                flag_equal = 1
                            else:
                                flag_equal = 0
                                break
                        if flag_equal == 1:  # 若该tuple中AV_c能够被其他规则确定,检测其是否满足规则
                            result2 = v[0]
                            if str(row[RHS]) == str(result2):
                                t_tc = t_tc + t_r
                            else:
                                t_tc = t_tc - t_r
                                print("匹配当前规则的tuple为：", row)
                                print("AV_c中",str(row[RHS]), "与", str(result2), "不符,对应的规则为", ruleinfo_c, "其置信度为", t_r)

                if t_tp==999:        #说明其中所有单元都无法被其他规则确定, 将其值重置为t0
                    t_tp=t0
                if t_tc < t_tp:
                    t_tuple = t_tc
                else:
                    t_tuple = t_tp


                # print("匹配该规则的部分为", AV_p, "-->",row[RHS],"其置信度为",t_tuple)
                if (str(row[RHS]) == str(result)):  # 该元组数据与规则相符合, 置信度增加
                    # print("此时t_rule=",t_rule,"t_tuple=",t_tuple,"math.ceil(math.log(1+t_tuple))=",math.ceil(math.log(1+t_tuple)))
                    # print("规则确定值为",result,";实际值为",row[RHS],"相符,规则置信度增加",t_rule, end='')
                    if t_tuple>0:
                        t_rule = t_rule + math.ceil(math.log(1+t_tuple))
                    else:
                        t_rule = t_rule + t_tuple
                    t_max = t_max
                    print("-->", t_rule, end='')
                else:  # 该元组数据与规则相违背, 计算对应tuple的置信度
                    # print("此时t_rule=", t_rule, "t_tuple=", t_tuple, "int(math.log(abs(t_tuple)))=",
                    #       int(math.log(abs(t_tuple))))
                    # print("规则确定值为", result, ";实际值为", row[RHS], "违反,规则置信度降低", t_rule, end='')
                    if t_tuple>0:
                        t_rule = t_rule - 2*t_tuple
                    else:
                        t_rule = t_rule + math.ceil(math.log(1+abs(t_tuple)))
                        print("-->", t_rule, end='')

                    if (t_rule < -100):
                        flag_trust = 0
                        return flag_trust  # 此时规则置信度过小,直接跳出循环,标记为错误

                if t_max < t_tuple:
                    t_max = t_tuple
                # if t_rule < t_max:
                #     flag_trust = 0
                #     return flag_trust

                # elif t_rule > t_max:
                #     error_row.append(row)
                #     rule_other.append(rule_p_name)


            print("最终规则置信度为",t_rule,"与其冲突的元组中置信度最大的为",t_max)
        if (t_rule > t_max ):
             flag_trust = 1  # 此时认为规则正确，修改数据
        elif (t_rule < t_max ):
             flag_trust = 0
             # print("最终规则置信度为", t_rule, "与其冲突的元组中置信度最大的为", t_max)
             return flag_trust  # 此时认为数据正确，修改规则
        self.rule[str(rulename)].update({'confidence': t_rule}) #规则置信度初始化可以考虑单拿出来
        print()
        return flag_trust


    def repair(self,iteration_num,path,order):

        print("执行了repair")
        f = open('data/save/att_name.txt', 'r')
        label2att = eval(f.read())
        f.close()
        # print(label2att)
        att2label = {v: k for k, v in label2att.items()}  #字典反向传递
        f = open('data/save/rules_final.txt', 'r')
        self.rule = eval(f.read())
        f.close()
        # print(self.rule)
        num = 0
        error_rule=0
        error_data=0
        conn = cx_Oracle.connect('system', 'Pjfpjf11', '127.0.0.1:1521/orcl')  # 连接数据库
        cursor = conn.cursor()
        for rulename, ruleinfo in list(self.rule.items()):
            num += 1
            print("修复第", num, "条规则及对应数据")
            # print("rulename:" + rulename)
            print("ruleinfo:", ruleinfo)

            left = list(ruleinfo['reason'].keys())
            # print(left)
            word = list(ruleinfo['reason'].values())
            # print(word)
            k = list(ruleinfo['result'].keys())
            right = k[0]
            v = list(ruleinfo['result'].values())
            result = v[0]

            LHS = []
            LHS.append(att2label[left[0]])
            RHS=att2label[right]
            sqlex = left[0] + "\"='" + word[0] + "'"
            i = 1
            # AV_p = "\""+left[0]+"\""+","     #把left里的数据转化为字符串形式
            while (i < len(left)):
                sqlex = sqlex + " and \"" + left[i] + "\"='" + word[i] + "'"
                # AV_p = AV_p +"\""+ left[i]+"\""+","
                LHS.append(att2label[left[i]])
                i += 1
                # print(sqlex)
            # print("AV_p索引：",LHS,"AV_c索引：",RHS)
            # AV_c = "\"" + right + "\""
            # print("AV_p:",AV_p,"AV_c:",AV_c)

            sql1 = "select * from \"" + path + "\" where \"" + sqlex
            #sql1 = "select " +AV_p+ AV_c + " from \"" + path + "\" where \"" + sqlex
            # sql1 = "select \"" + right + "\" from \"" + path + "\" where \"" + sqlex
            print(sql1)
            cursor.execute(sql1)  # "City","State" ,where rownum<=10
            rows = cursor.fetchall()
            # print("rows:")
            flag_trust=self.detect(rows,result,rulename,LHS,RHS,att2label,label2att)

            if (flag_trust == 3):       # 3代表规则正确, 且无任何冲突, 直接进行下一条规则
                continue

            if (flag_trust == 0):
                error_rule += 1

            s1=0
            while (flag_trust == 0 and s1 < 3):
                print("规则不可信，修复规则")
                print("修复规则右侧")
                s1 += 1
                result=self.multipredict_rules_probability(word)
                print("右侧更改为",result)
                flag_trust=self.detect(rows,result,rulename,LHS,RHS,att2label,label2att)

                    # print("trust=",trust)
                if (flag_trust==1):
                    print("规则修复成功")
                    addtwodimdict(self.rule,str(rulename), 'result', {str(right): str(result)})
                    print("修改后规则为",self.rule[str(rulename)])
                elif (flag_trust==0 and s1==5):
                    print("规则右侧无可替换修复")


            s2 = 0
            while (flag_trust == 0 and s2 < 3):
                result = v[0]
                print("修复规则左侧")
                s2 += 1
                min=10
                flag = int(att2label[left[0]])
                # print(flag)
                if (min > flag):
                    min = flag  # 目前reason部分最左侧对应的索引
                # print(min)
                if(min==0):
                    print("规则左侧无可增加修复，删除该条规则")
                    del self.rule[str(rulename)]
                    break
                left_new=label2att[min-1]
                print("增加",left_new,"信息")
                sqladd= "select \"" + left_new + "\" from \"" + path + "\" where \"" + sqlex+"and \"" + right + "\"='" + result + "'"
                print("sqladd:",sqladd)
                cursor.execute(sqladd)
                rows_left = cursor.fetchall()
                # print(rows[0][0])
                # print(word)
                # print(self.rule[str(rulename)])


                #重构字典
                if(rows_left ==[]):
                    # print("规则左侧无满足条件修改,删除该条规则")
                    del self.rule[str(rulename)]
                    break
                # print(rows_left)
                addtwodimdict(self.rule[str(rulename)], 'reason', str(left_new), str(rows_left[0][0]))
                for n in range(len(word)):
                    del self.rule[str(rulename)]['reason'][left[n]]
                    addtwodimdict(self.rule[str(rulename)], 'reason', str(left[n]), str(word[n]))
                # left = list(ruleinfo['reason'].keys())
                ######否则，字典里新增的内容应该在最前面，但现在在最后面
                # tex=[]
                # tex.append(rows_left[0][0])
                # for t in range(len(word)):
                #     tex.append(word[t])
                # # print(tex)
                # word = tex
                left = list(ruleinfo['reason'].keys())
                word = list(ruleinfo['reason'].values())
                # print(word)
                # print(self.rule[str(rulename)])
                sqlex = left[0] + "\"='" + word[0] + "'"
                i = 1
                while (i < len(left)):
                    sqlex = sqlex + " and \"" + left[i] + "\"='" + word[i] + "'"
                    i += 1
                sql1 = "select * from \"" + path + "\" where \"" + sqlex
                # print(sql1)
                cursor.execute(sql1)  # "City","State" ,where rownum<=10
                rows = cursor.fetchall()

                if (len(rows)<3):
                    continue

                result = self.multipredict_rules_argmax(word)
                # print(result)
                flag_trust=self.detect(rows,result,rulename,LHS,RHS,att2label,label2att)
                if (flag_trust == 1):
                    print("规则修复成功")
                    print("修改后规则为", self.rule[str(rulename)])
                elif (flag_trust == 1 and min!=0) :
                    # print("规则左侧无满足条件修改,删除该条规则")
                    del self.rule[str(rulename)]
                    break
            if (flag_trust == 0):
                print("规则无可用修复,删除该规则")

            if (flag_trust == 1):
                t0=1
                for row in rows:
                    if (str(row[RHS]) == str(result)):
                        continue
                    else:
                        AV_p = []
                        t_tp = 999  # 当前tuple的置信度, 计算为一个tuple中不同AV_i中的置信度最小值,为了避免初始值干扰,先设定个大值
                        t_tc = t0
                        flag_p=0     #用以记录AV_p中置信度最小的对应的属性位置
                        rule_p_name=[] #用以记录能够修复上述的AV_p中置信度最小的对应的属性的具有最大置信度的规则
                        print("匹配当前规则的tuple为：", row)
                        for i in LHS:  # 计算一个tuple中不同AV_i中的置信度最小值
                            AV_p.append(row[i])
                            t_AV_i = t0
                            attribute_p = label2att[i]
                            rulename_p_max = []
                            t_rmax = -999       # 下面遍历的字典中能纠正AV_i的规则中最大置信度, 初始设为极小值
                            for rulename_p, ruleinfo_p in list(self.rule.items()):  # 遍历字典
                                if rulename == rulename_p:
                                    continue
                                if t_AV_i > 100 or t_AV_i < -100:
                                    break
                                v = list(ruleinfo_p['result'].values())
                                left = list(ruleinfo_p['reason'].keys())
                                word = list(ruleinfo_p['reason'].values())
                                k = list(ruleinfo_p['result'].keys())
                                t_r = ruleinfo_p['confidence']
                                if t_r < 0:
                                    continue
                                right = k[0]
                                if attribute_p == right:
                                    flag_equal = 0  # 规则能否确定row[i]的标记
                                    for k in range(len(left)):
                                        if row[att2label[left[k]]] == word[k]:  # 若row[i]所在的tuple满足某条规则的全部AV_p,标记为1
                                            flag_equal = 1
                                        else:
                                            flag_equal = 0
                                            break
                                    if flag_equal == 1:  # 若该tuple中row[i]能够被其他规则确定,检测其是否满足规则
                                        # print(row, "中")
                                        # print(right, "可以由其他规则确定：", ruleinfo)
                                        result2 = v[0]
                                        if t_rmax < t_r:  # 记录这些规则中最大的规则置信度
                                            t_rmax = t_rmax
                                            rulename_p_max = rulename_p  # 记录该最可信规则在字典中的标识
                                        if str(row[i]) == str(result2):
                                            t_AV_i = t_AV_i + t_r
                                        else:
                                            t_AV_i = t_AV_i - t_r
                                            print("AV_p中", str(row[i]), "与", str(result2), "不符,对应的规则为", ruleinfo_p,
                                                  "其置信度为", t_r)

                            if t_tp > t_AV_i:
                                t_tp = t_AV_i
                                flag_p=i                    #记录置信度最小的AV_i的索引
                                rule_p_name=rulename_p_max  #记录能纠正该AV_i的置信度最大的规则名

                        for rulename_c, ruleinfo_c in list(self.rule.items()):  # 遍历字典,计算t_c
                            if rulename == rulename_c:
                                continue
                            v = list(ruleinfo_c['result'].values())
                            left = list(ruleinfo_c['reason'].keys())
                            word = list(ruleinfo_c['reason'].values())
                            k = list(ruleinfo_c['result'].keys())
                            t_r = ruleinfo_c['confidence']
                            if t_r < 0:
                                continue
                            right = k[0]
                            attribute_c = label2att[RHS]
                            if attribute_c == right:
                                flag_equal = 0  # 规则能否确定row[i]的标记
                                for k in range(len(left)):
                                    if row[att2label[left[k]]] == word[k]:  # 若AV_c所在的tuple满足某条规则的全部AV_p,标记为1
                                        flag_equal = 1
                                    else:
                                        flag_equal = 0
                                        break
                                if flag_equal == 1:  # 若该tuple中AV_c能够被其他规则确定,检测其是否满足规则
                                    result2 = v[0]
                                    if str(row[RHS]) == str(result2):
                                        t_tc = t_tc + t_r
                                    else:
                                        t_tc = t_tc - t_r
                                        print("AV_c中", str(row[RHS]), "与", str(result2), "不符,对应的规则为", ruleinfo_c, "其置信度为",
                                              t_r)

                        if t_tp == 999:  # 说明其中所有单元都无法被其他规则确定, 将其值重置为t0
                            t_tp = t0
                        if t_tc < t_tp or t_tc == t_tp:
                            print("此时认为数据结果部分错误,根据规则修复数据,当前规则为",rulename,"-->",result,"t_p为",t_tp,"t_c为",t_tc)
                            for x in range(len(row)-1):  # t2
                                if x == 0:
                                    sql_info = "\"" + label2att[x] + "\"='" + row[x] + "'"
                                else:
                                    sql_info = sql_info + " and \"" + label2att[x] + "\"='" + row[x] + "'"
                            sql_update = "update \"" + path + "\" set \"Label\"='2' , \"" + label2att[RHS] + "\"='" + result + "' where  " + sql_info + ""
                            print("原始：", sql_info)
                            print("Update信息：", sql_update)
                            cursor.execute(sql_update)
                            conn.commit()
                        else:
                            print(rule_p_name)
                            if rule_p_name==[]:
                                print("可能有错误")
                                continue
                            rname=self.rule[str(rule_p_name)]
                            v2 = list(rname['result'].values())
                            result2 = v2[0]
                            print("此时认为数据推论部分错误,根据规则修复数据,当前规则为", rule_p_name, "-->", result2, "t_p为", t_tp, "t_c为", t_tc)
                            for x in range(len(row)-1):  # t2
                                if x == 0:
                                    sql_info = "\"" + label2att[x] + "\"='" + row[x] + "'"
                                else:
                                    sql_info = sql_info + " and \"" + label2att[x] + "\"='" + row[x] + "'"
                            sql_update = "update \"" + path + "\" set \"Label\"='2' , \"" + label2att[flag_p] + "\"='" + result2 + "' where  " + sql_info + ""
                            print("原始：", sql_info)
                            print("Update信息：", sql_update)
                            cursor.execute(sql_update)
                            conn.commit()
                            continue


            # if (flag_trust == 1):
            #     # 只修复标记为'1'的错误数据，并标记为2
            #     # sql_update = "update \"Hosp2_rule_copy\" set \"" + right + "\"='" + result + "' , \"Label\"='2'   where \"Label\"='1' or \"Label\"='2' and \"" + sqlex
            #     # print(sql_update)
            #     # sql_check = "select * from \"" + path + "\"   where  \"" + sqlex#\"" + right + "\"  #(\"Label\"='1' or \"Label\"='2') and
            #     sql_check = "select * from \"" + path + "\"   where  (\"Label\"='1' or \"Label\"='2') and \"" + sqlex  # \"" + right + "\"
            #     # print(sql_check)
            #     cursor.execute(sql_check)
            #     row_check = cursor.fetchall()
            #     row_check = [x[:-1] for x in row_check]
            #     if order == 0:
            #         row_check = [x[::-1] for x in row_check]
            #     # print(row_check)
            #     flag_check=att2label[right]
            #     for row in row_check:
            #         t2 = len(row)
            #         att = list(label2att.values())
            #         # print(right,row[flag_check],result)
            #         if (row[flag_check]!=result):
            #             error_data+=1
            #             for i in range(t2):  # t2
            #                 if i == 0:
            #                     sql_info = "\"" + att[i] + "\"='" + row[i] + "'"
            #                 else:
            #                     sql_info = sql_info + " and \"" + att[i] + "\"='" + row[i] + "'"
            #             # sql_info = sql_info + " and (\"Label\"='1' or \"Label\"='2')"
            #             print("原始：",sql_info)
            #             # row = list(row)
            #             # row[flag_check] = result
            #             # print("row[flag_check]:",row[flag_check])
            #             # print("result",result)
            #             sql_update="update \"" + path + "\" set \"Label\"='2' , \"" + att[flag_check] + "\"='" + result + "' where  " + sql_info + ""
            #             print("Update信息：", sql_update)
            #             cursor.execute(sql_update)
            #             conn.commit()

        cursor.close()
        conn.close()
            # if num>200:
            #     break

        print("修复完成")
        print("保存修复规则")
        print("规则字典大小", len(self.rule))
        # print(str(self.rule))
        f = open('data/save/rules_final.txt', 'w')
        f.write(str(self.rule))
        f.close()
        with open('data/save/log.txt', 'a') as f:
            f.write("本次共使用规则数量")
            f.write(str(num))
            f.write("规则错误数量")
            f.write(str(error_rule))
            f.write("数据错误数量")
            f.write(str(error_data))
            f.write("__________")
            f.close()


        # 这里可用来循环修复直到无新错误数据
        # if (iteration_num>0):
        #     print(iteration_num)
        #     if (error_rule != 0):
        #         self.repair(iteration_num-1,path,order)




    def save(self, path):
        weights = []
        for layer in self.layers:
            w = layer.get_weights()
            weights.append(w)
        with open(path, 'wb') as f:
            pickle.dump(weights, f)

    def load(self, path):
        with open(path, 'rb') as f:
            weights = pickle.load(f)
        for layer, w in zip(self.layers, weights):
            layer.set_weights(w)

def Discriminator(V, E, H=64, dropout=0.1):
    '''
    Disciriminator model.
    # Arguments:
        V: int, Vocabrary size
        E: int, Embedding size
        H: int, LSTM hidden size
        dropout: float
    # Returns:
        discriminator: keras model
            input: word ids, shape = (B, T)
            output: probability of true data or not, shape = (B, 1)
    '''
    input = Input(shape=(None,), dtype='int32', name='Input')   # (B, T)
    out = Embedding(V, E, mask_zero=True, name='Embedding')(input)  # (B, T, E)
    out = LSTM(H)(out)
    out = Highway(out, num_layers=1)
    out = Dropout(dropout, name='Dropout')(out)
    out = Dense(1, activation='sigmoid', name='FC')(out)

    discriminator = Model(input, out)
    return discriminator

def Highway(x, num_layers=1, activation='relu', name_prefix=''):
    '''
    Layer wrapper function for Highway network
    # Arguments:
        x: tensor, shape = (B, input_size)
    # Optional Arguments:
        num_layers: int, dafault is 1, the number of Highway network layers
        activation: keras activation, default is 'relu'
        name_prefix: str, default is '', layer name prefix
    # Returns:
        out: tensor, shape = (B, input_size)
    '''
    input_size = K.int_shape(x)[1]
    for i in range(num_layers):
        gate_ratio_name = '{}Highway/Gate_ratio_{}'.format(name_prefix, i)
        fc_name = '{}Highway/FC_{}'.format(name_prefix, i)
        gate_name = '{}Highway/Gate_{}'.format(name_prefix, i)

        gate_ratio = Dense(input_size, activation='sigmoid', name=gate_ratio_name)(x)
        fc = Dense(input_size, activation=activation, name=fc_name)(x)
        x = Lambda(lambda args: args[0] * args[2] + args[1] * (1 - args[2]), name=gate_name)([fc, x, gate_ratio])
    return x

def addtwodimdict(thedict, key_a, key_b, val):
  if key_a in thedict:
    thedict[key_a].update({key_b: val})
  else:
    thedict.update({key_a:{key_b: val}})
