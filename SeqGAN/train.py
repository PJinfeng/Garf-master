from SeqGAN.models import GeneratorPretraining, Discriminator, Generator
from SeqGAN.utils import GeneratorPretrainingGenerator, DiscriminatorGenerator
from SeqGAN.rl import Agent, Environment
from keras.optimizers import Adam
import os
import numpy as np
import tensorflow as tf
sess = tf.Session()
import keras.backend as K
K.set_session(sess)

import code

class Trainer(object):
    '''
    Manage training
    '''
    def __init__(self, order, B, T, g_E, g_H, d_E, d_H, d_dropout, generate_samples,path_pos, path_neg, path_rules, g_lr=1e-3, d_lr=1e-3, n_sample=16,  init_eps=0.1):
        self.B, self.T = B, T               #batch size，max_length
        self.order=order
        self.g_E, self.g_H = g_E, g_H
        self.d_E, self.d_H = d_E, d_H
        self.d_dropout = d_dropout
        self.generate_samples = generate_samples
        self.g_lr, self.d_lr = g_lr, d_lr
        self.eps = init_eps
        self.init_eps = init_eps        #探索率ϵ。即策略是以1−ϵ的概率选择当前最大价值的动作，以ϵ的概率随机选择新动作
        self.top = os.getcwd()          #os.getcwd() 方法用于返回当前工作目录
        self.path_pos = path_pos        #原始数据所在地址
        self.path_neg = path_neg        #生成数据所在地址
        self.path_rules = path_rules
        
        self.g_data = GeneratorPretrainingGenerator(self.path_pos, order=order, B=B, T=T, min_count=1) # next方法产生x, y_true数据; 都是同一个数据，比如[BOS, 8, 10, 6, 3, EOS]，[8, 10, 6, 3, EOS]
        self.d_data = DiscriminatorGenerator(path_pos=self.path_pos, order=order, path_neg=self.path_neg, B=self.B, shuffle=True) # next方法产生 pos数据和neg数据

        self.V = self.g_data.V          #数据中词汇总量
        self.agent = Agent(sess, B, self.V, g_E, g_H, g_lr)
        self.g_beta = Agent(sess, B, self.V, g_E, g_H, g_lr)

        self.discriminator = Discriminator(self.V, d_E, d_H, d_dropout)

        self.env = Environment(self.discriminator, self.g_data, self.g_beta, n_sample=n_sample)

        self.generator_pre = GeneratorPretraining(self.V, g_E, g_H)         #一个4层的神经网络input-embedding-lstm-dense

        self.rule={}

    def pre_train(self, g_epochs=3, d_epochs=1, g_pre_path=None ,d_pre_path=None, g_lr=1e-3, d_lr=1e-3):        #实际参数由config给出
        self.pre_train_generator(g_epochs=g_epochs, g_pre_path=g_pre_path, lr=g_lr)

        self.pre_train_discriminator(d_epochs=d_epochs, d_pre_path=d_pre_path, lr=d_lr)

    def pre_train_generator(self, g_epochs=3, g_pre_path=None, lr=1e-3):
        print("预训练生成器")
        if g_pre_path is None:
            self.g_pre_path = os.path.join(self.top, 'data', 'save', 'generator_pre.hdf5')  #D:\PycharmProjects\Garf-master\data\save\generator_pre.hdf5
        else:
            self.g_pre_path = g_pre_path

        g_adam = Adam(lr)
        self.generator_pre.compile(g_adam, 'categorical_crossentropy')  #进行训练，优化器为Adam，损失函数为分类交叉熵函数，适用于多分类
        print('Generator pre-training')
        self.generator_pre.summary()            #keras中model.summary()用于输出模型各层的参数状况
        # print("++++++++++++++++++++")
        self.generator_pre.fit_generator(       #返回值为一个 History 对象。其 History.history 属性是连续 epoch 训练损失和评估值，以及验证集损失和评估值的记录
            self.g_data,                        #此处应为一个生成器或 Sequence (keras.utils.Sequence) 对象的实例
            steps_per_epoch=None,
            epochs=g_epochs)
        self.generator_pre.save_weights(self.g_pre_path)    #保存权重到generator_pre.hdf5
        self.reflect_pre_train()                #将layer层权重映射给agent

    def pre_train_discriminator(self, d_epochs=1, d_pre_path=None, lr=1e-3):
        print("预训练判别器")
        if d_pre_path is None:
            self.d_pre_path = os.path.join(self.top, 'data', 'save', 'discriminator_pre.hdf5')
        else:
            self.d_pre_path = d_pre_path

        print('Start Generating sentences')
        self.agent.generator.generate_samples(self.T, self.g_data,
            self.generate_samples, self.path_neg)      #生成器生成序列，写入output_file位置的txt

        self.d_data = DiscriminatorGenerator(
            path_pos=self.path_pos,             #真实数据的采样
            order=self.order,
            path_neg=self.path_neg,             #读取刚刚生成器生成的数据
            B=self.B,
            shuffle=True)

        d_adam = Adam(lr)
        self.discriminator.compile(d_adam, 'binary_crossentropy')
        self.discriminator.summary()
        print('Discriminator pre-training')

        self.discriminator.fit_generator(
            self.d_data,
            steps_per_epoch=None,
            epochs=d_epochs)
        self.discriminator.save(self.d_pre_path)

    def load_pre_train(self, g_pre_path, d_pre_path):
        self.generator_pre.load_weights(g_pre_path)
        self.reflect_pre_train()
        self.discriminator.load_weights(d_pre_path)

    def load_pre_train_g(self, g_pre_path):
        self.generator_pre.load_weights(g_pre_path)
        self.reflect_pre_train()

    def load_pre_train_d(self, d_pre_path):
        self.discriminator.load_weights(d_pre_path)

    def reflect_pre_train(self):                        #将layer层权重映射给agent
        i = 0
        for layer in self.generator_pre.layers:
            if len(layer.get_weights()) != 0:           #若该层权重不为0
                w = layer.get_weights()
                self.agent.generator.layers[i].set_weights(w)       #则将agent中对应层的权重置为w
                self.g_beta.generator.layers[i].set_weights(w)
                i += 1

    def train(self, steps=10, g_steps=1, d_steps=1, d_epochs=1,
        g_weights_path='data/save/generator.pkl',
        d_weights_path='data/save/discriminator.hdf5',
        verbose=True,
        head=1):
        print("开始正式训练")
        d_adam = Adam(self.d_lr)
        self.discriminator.compile(d_adam, 'binary_crossentropy')
        self.eps = self.init_eps
        for step in range(steps):
            print("当前整体回合数",step+1)
            # Generator training
            for _ in range(g_steps):
                print("G-step")
                rewards = np.zeros([self.B, self.T])                    #reward建立空表
                self.agent.reset()                                      #agent重置
                self.env.reset()                                        #env重置
                for t in range(self.T):                                 #开始迭代地训练生成器
                    state = self.env.get_state()

                    action = self.agent.act(state, epsilon=0.0)

                    _next_state, reward, is_episode_end, _info = self.env.step(action)
                    self.agent.generator.update(state, action, reward)
                    rewards[:, t] = reward.reshape([self.B, ])
                    if is_episode_end:
                        if verbose:
                            print('Reward: {:.3f}, Episode end'.format(np.average(rewards)))
                            self.env.render(head=head)
                        break
            #print("flag")
            # Discriminator training
            for _ in range(d_steps):
                print("D-step")
                self.agent.generator.generate_samples(
                    self.T,
                    self.g_data,
                    self.generate_samples,
                    self.path_neg)
                self.d_data = DiscriminatorGenerator(
                    path_pos=self.path_pos,
                    order=self.order,
                    path_neg=self.path_neg,
                    B=self.B,
                    shuffle=True)
                self.discriminator.fit_generator(
                    self.d_data,
                    steps_per_epoch=None,
                    epochs=d_epochs)

            # Update env.g_beta to agent
            self.agent.save(g_weights_path)
            self.g_beta.load(g_weights_path)

            self.discriminator.save(d_weights_path)
            self.eps = max(self.eps*(1- float(step) / steps * 4), 1e-4)

    def save(self, g_path, d_path):
        self.agent.save(g_path)
        self.discriminator.save(d_path)

    def load(self, g_path, d_path):
        self.agent.load(g_path)
        self.g_beta.load(g_path)
        self.discriminator.load_weights(d_path)




    def generate_rules(self, file_name, generate_samples):
        #self.B=1
        path_rules = os.path.join(self.top, 'data', 'save', file_name)
        print(path_rules)

        self.agent.generator.generate_rules(
            8, self.g_data, generate_samples, path_rules)

    # def predict_rules(self):
    #     # self.agent.generator.predict_rules()
    #     result=self.agent.generator.multipredict_rules_argmax(reason=['10005','BOAZ','AL','36251'])
    #     result_ = self.agent.generator.multipredict_rules_argmax(reason=['10005', 'BOAZ', 'AL', '36251'])


    def train_rules(self,rule_len,path):
        path_rules = os.path.join(self.top, 'data', 'save', path)
        self.agent.generator.train_rules(rule_len,path_rules)

    def filter(self,path):
        self.agent.generator.filter(path)

    def repair(self,path):
        self.agent.generator.repair(1,path,self.order)#3

    # def repair_SeqGAN(self):
    #     self.agent.generator.repair_SeqGAN()