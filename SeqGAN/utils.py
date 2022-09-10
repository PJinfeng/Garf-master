import numpy as np
import random
import linecache
from keras.utils import Sequence
from keras.utils.np_utils import to_categorical
import cx_Oracle

import code

class Vocab:    #建立词汇库
    def __init__(self, word2id, unk_token):
        self.word2id = dict(word2id)                            #建立一个字典
        self.id2word = {v: k for k, v in self.word2id.items()}  #将字典反向传递给id2word
        self.unk_token = unk_token

    def build_vocab(self, sentences, min_count=1):
        word_counter = {}
        for sentence in sentences:
            for word in sentence:
                word_counter[word] = word_counter.get(word, 0) + 1  #字典类型赋值，其中.get(word,0)+1 是对word出现的频率进行统计

        for word, count in sorted(word_counter.items(), key=lambda x: -x[1]):    #sorted() 函数对所有可迭代的对象进行排序操作,按频率出现的次数从多到少排列
            if count < min_count:
                break
            _id = len(self.word2id)     #当前字典大小
            self.word2id.setdefault(word, _id)  #返回字典中word对应的值，即当前句子中word的出现次数，若不存在则返回_id
            self.id2word[_id] = word

        self.raw_vocab = {w: word_counter[w] for w in self.word2id.keys() if w in word_counter} #字典合集{每个词：对应id}

    def sentence_to_ids(self, sentence):
        return [self.word2id[word] if word in self.word2id else self.word2id[self.unk_token] for word in sentence]

def load_data(path, order):

    conn = cx_Oracle.connect('system', 'Pjfpjf11', '127.0.0.1:1521/orcl')  # 连接数据库
    cursor = conn.cursor()
    sql1 = "select * from \"" + path + "\" "
    print(sql1)
    cursor.execute(sql1)  # "City","State" ,where rownum<=10
    rows = cursor.fetchall()
    rows = [x[:-1] for x in rows]
    # print(rows)

    if order == 1:
        print("正序加载数据……")

    elif order == 0:
        print("逆序加载数据……")
        rows = [x[::-1] for x in rows]
        # print(rows)
    cursor.close()
    conn.close()

    return rows

def sentence_to_ids(vocab, sentence, UNK=3):
    '''
    # Arguments:
        vocab: SeqGAN.utils.Vocab
        sentence: list of str
    # Returns:
        ids: list of int
    '''
    ids = [vocab.word2id.get(word, UNK) for word in sentence]
    return ids

def pad_seq(seq, max_length, PAD=0):                    #句子长度未到25的，后面补0
    """
    :param seq: list of int,
    :param max_length: int,
    :return seq: list of int,
    """
    seq += [PAD for i in range(max_length - len(seq))]
    return seq

def print_ids(ids, vocab, verbose=True, exclude_mark=True, PAD=0, BOS=1, EOS=2):
    '''
    :param ids: list of int,
    :param vocab:
    :param verbose(optional): 
    :return sentence: list of str
    '''
    sentence = []
    for i, id in enumerate(ids):
        word = vocab.id2word[id]
        if exclude_mark and id == EOS:
            break
        if exclude_mark and id in (BOS, PAD):
            continue
        sentence.append(sentence)
    if verbose:
        print(sentence)
    return sentence


class GeneratorPretrainingGenerator(Sequence):              #直接从原数据中取数据，制作x和y_true，作为x_train 和 y_train即训练数据和标签
    def __init__(self, path, order, B, T, min_count=1, shuffle=True):
        self.PAD = 0
        self.BOS = 1
        self.EOS = 2
        self.UNK = 3
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.BOS_TOKEN = '<S>'
        self.EOS_TOKEN = '</S>'
        self.path = path
        self.B = B
        self.T = T
        self.min_count = min_count
        self.count = 0

        sentences = load_data(path, order)

        print("原始数据", sentences)
        self.rows = sentences

        default_dict = {
            self.PAD_TOKEN: self.PAD,
            self.BOS_TOKEN: self.BOS,
            self.EOS_TOKEN: self.EOS,
            self.UNK_TOKEN: self.UNK,
        }
        self.vocab = Vocab(default_dict, self.UNK_TOKEN)

        self.vocab.build_vocab(sentences, self.min_count)

        self.word2id = self.vocab.word2id
        self.id2word = self.vocab.id2word
        self.raw_vocab = self.vocab.raw_vocab
        self.V = len(self.vocab.word2id)
        # with open(path, 'r', encoding='utf-8') as f:
        #     self.n_data = sum(1 for line in f)          #数据行数
        self.n_data = len(sentences)                  # 原始数据行数
        self.word2id = self.vocab.word2id
        self.id2word = self.vocab.id2word
        f = open('data/save/word2id.txt', 'w')
        f.write(str(self.word2id))
        f.close()
        f = open('data/save/id2word.txt', 'w')
        f.write(str(self.id2word))
        f.close()
        print("+++++++")
        #记录f.read().lower()是不区分大小写；chars = sorted(list(set(raw_text)))能拆成字符级排序，其中set去重，sorted排序
        self.shuffle = shuffle
        self.idx = 0
        self.len = self.__len__()
        
        self.reset()


    def __len__(self):
        return self.n_data // self.B        #总数据数除以一次训练所选取的样本数

    def __getitem__(self, idx):             ##读取原始数据中的第idx行，生成x和y_true
        '''
        Get generator pretraining data batch.
        # Arguments:
            idx: int, index of batch
        # Returns:
            None: no input is needed for generator pretraining.
            x: numpy.array, shape = (B, max_length)
            y_true: numpy.array, shape = (B, max_length, V)
                labels with one-hot encoding.
                max_length is the max length of sequence in the batch.
                if length smaller than max_length, the data will be padded.
        '''
        # print("****************************")
        # self.count =self.count+1
        # print(self.count)
        # print("idx为",idx)

        x, y_true = [], []
        start = (idx-1) * self.B + 1
        end = idx * self.B + 1
        max_length = 0
        for i in range(start, end):
            if self.shuffle:
                # print("self.shuffle:",self.shuffle)
                idx = self.shuffled_indices[i]
            else:
                # print("shuffle=False")
                idx = i
            # sentence = linecache.getline(self.path, idx)    #读取原始数据中的第idx行
            # words = sentence.strip().split()
            # print(idx)
            sentence = self.rows[idx]                         #读取查询结果中的第idx行
            words = []
            for i in sentence:
                words.append(i)
            ids = sentence_to_ids(self.vocab, words)        #ids是一个list,存放的是原始数据中word的id序列

            ids_x, ids_y_true = [], []                      #置空

            ids_x.append(self.BOS)                          #开头写入标识符BOS
            ids_x.extend(ids)                               #添加ids即，当前句子中多个word所对应的id
            ids_x.append(self.EOS) # ex. [BOS, 8, 10, 6, 3, EOS]
            x.append(ids_x)                                 #X为多个句子组合而成的列表,np.array(x)).shape=(B,)，即B行，每行1个元素
            # print("x:",x)
            # print(type(x))
            # print((np.array(x)).shape)

            ids_y_true.extend(ids)
            ids_y_true.append(self.EOS) # ex. [8, 10, 6, 3, EOS]
            y_true.append(ids_y_true)                       #截至目前，y_true与x，数据和形状都相似，都是(B,)，只是其中每个元素都少一个开头的BOS
            # print("y_true:",y_true)
            # print("y_true:")
            # print(type(y_true))
            # print((np.array(y_true)).shape)


            max_length = max(max_length, len(ids_x))


        if self.T is not None:
            max_length = self.T

        for i, ids in enumerate(x):
            x[i] = x[i][:max_length]                #循环了len(X)次，X[i]为列表X中的第i个句子,并截断到max_length的长度

        for i, ids in enumerate(y_true):
            y_true[i] = y_true[i][:max_length]

        x = [pad_seq(sen, max_length) for sen in x]     #句子长度未到25的，后面补0
        x = np.array(x, dtype=np.int32)

        y_true = [pad_seq(sen, max_length) for sen in y_true]
        y_true = np.array(y_true, dtype=np.int32)
        # print("y_true:", y_true[0][0])
        y_true = to_categorical(y_true, num_classes=self.V)     #将原有的类别向量转换为one-hot的形式,维度为总词汇数量
        # print("x:", x[0])
        # print("y_true转化后:",y_true[0][0])
        # print("x:", x)
        # print("y_true:",y_true)

        return (x, y_true)

    def next(self):
        if self.idx >= self.len:
            self.reset()
            raise StopIteration
        x, y_true = self.__getitem__(self.idx)
        self.idx += 1
        return (x, y_true)

    def reset(self):                                    #重置，重新生成一个大小为n_data的乱序数组
        self.idx = 0
        if self.shuffle:
            self.shuffled_indices = np.arange(self.n_data)
            random.shuffle(self.shuffled_indices)           #将列表中元素打乱
        #print(self.shuffled_indices)                       #乱序的大小为n_data的乱序数组，[3850 1111 4587 ... 2454 3013 3144]

    def on_epoch_end(self):
        self.reset()
        pass

    def __iter__(self):
        return self

class DiscriminatorGenerator(Sequence):
    '''
    Generate generator pretraining data.
    # Arguments
        path_pos: str, path to true data
        path_neg: str, path to generated data
        B: int, batch size
        T (optional): int or None, default is None.
            if int, T is the max length of sequential data.
        min_count (optional): int, minimum of word frequency for building vocabrary
        shuffle (optional): bool

    # Params
        PAD, BOS, EOS, UNK: int, id
        PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN: str
        B, min_count: int
        vocab: Vocab
        word2id: Vocab.word2id
        id2word: Vocab.id2word
        raw_vocab: Vocab.raw_vocab
        V: the size of vocab
        n_data: the number of rows of data

    # Examples
        generator = VAESequenceGenerator('./data/train_x.txt', 32)
        X, Y = generator.__getitem__(idx=11)
        print(X[0])
        >>> 8, 10, 6, 3, 2, 0, 0, ..., 0
        print(Y)
        >>> 0, 1, 1, 0, 1, 0, 0, ..., 1

        id2word = generator.id2word

        x_words = [id2word[id] for id in X[0]]
        print(x_words)
        >>> I have a <UNK> </S> <PAD> ... <PAD>
    '''
    def __init__(self, path_pos, order, path_neg, B, T=40, min_count=1, shuffle=True):
        self.PAD = 0
        self.BOS = 1
        self.EOS = 2
        self.UNK = 3
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.BOS_TOKEN = '<S>'
        self.EOS_TOKEN = '</S>'
        self.path_pos = path_pos
        self.path_neg = path_neg
        self.B = B
        self.T = T
        self.min_count = min_count

        sentences = load_data(path_pos, order)
        self.rows=sentences

        default_dict = {
            self.PAD_TOKEN: self.PAD,
            self.BOS_TOKEN: self.BOS,
            self.EOS_TOKEN: self.EOS,
            self.UNK_TOKEN: self.UNK,
        }
        self.vocab = Vocab(default_dict, self.UNK_TOKEN)

        self.vocab.build_vocab(sentences, self.min_count)

        self.word2id = self.vocab.word2id
        self.id2word = self.vocab.id2word
        self.raw_vocab = self.vocab.raw_vocab
        self.V = len(self.vocab.word2id)


        # with open(path_pos, 'r', encoding='utf-8') as f:
        #     self.n_data_pos = sum(1 for line in f)              #原始数据行数

        self.n_data_pos = len(self.rows)                             #原始数据行数
        with open(path_neg, 'r', encoding='utf-8') as f:
            self.n_data_neg = sum(1 for line in f)              #生成数据行数
        # f = open('data/save/word2id-d.txt', 'w')
        # f.write(str(self.word2id))
        # f.close()

        self.n_data = self.n_data_pos + self.n_data_neg
        self.shuffle = shuffle
        self.idx = 0
        self.len = self.__len__()
        self.reset()

    def __len__(self):
        return self.n_data // self.B

    def __getitem__(self, idx):
        '''
        Get generator pretraining data batch.
        # Arguments:
            idx: int, index of batch
        # Returns:
            X: numpy.array, shape = (B, max_length)
            Y: numpy.array, shape = (B, ) ,label:true=1,generated data=0
        '''
        X, Y = [], []
        start = (idx-1) * self.B + 1
        end = idx * self.B + 1
        max_length = 0

        for i in range(start, end):
            # print(start)
            # print(end)
            # print("前：",idx)
            idx = self.indicies[i]    #在原始数据与生成数据索引中随机选取一个值
            # print("后",idx)
            is_pos = 1
            if idx < 0:
                is_pos = 0
                idx = -1 * idx
            idx = idx - 1

            if is_pos == 1:
                # sentence = linecache.getline(self.path_pos, idx) # str  #读取原始数据中的第idx行
                sentence = self.rows[idx]
                words = []
                for i in sentence:
                    words.append(i)
            elif is_pos == 0:
                sentence = linecache.getline(self.path_neg, idx) # str  #读取生成数据中的第idx行
                words = sentence.strip().split()
            # words = sentence.strip().split()  # list of str  ex.['"261318"', '"SALEM"', '"MO"', '"65560"', '"DENT"', '"5737296626"', '"Pregnancy', 'and', 'Delivery', 'Care"', '"PC_01"', '"Elective', 'Delivery"']
            # print("word:",words)
            ids = sentence_to_ids(self.vocab, words) # list of ids ex.[1261, 1262, 51, 1263, 1264, 1265, 136, 31, 137, 27, 138, 139, 140]
            # print("ids:",ids)

            x = []
            x.extend(ids)
            x.append(self.EOS) # ex. [8, 10, 6, 3, EOS]
            X.append(x)                             #句子合集，ex.[[703, 250, 52, 704, 250, 705, 71, 27, 72, 73, 74, 8, 75, 76, 31, 77, 78, 2], [421, 422, 9, 423, 231, 424, 42, 27, 89, 90, 91, 92, 93, 94, 2]]
            Y.append(is_pos)

            max_length = max(max_length, len(x))

        if self.T is not None:
            max_length = self.T

        for i, ids in enumerate(X):
            X[i] = X[i][:max_length]                #去掉超过最大长度的部分

        X = [pad_seq(sen, max_length) for sen in X] #当前部分结束到最大长度部分补0
        X = np.array(X, dtype=np.int32)
        # print("X:",X)

        return (X, Y)

    def next(self):
        # print(self.idx)
        if self.idx >= self.len:
            self.reset()
            raise StopIteration
        X, Y = self.__getitem__(self.idx)
        self.idx += 1
        # print(X)
        # print(Y)
        return (X, Y)

    def reset(self):
        self.idx = 0
        pos_indices = np.arange(start=1, stop=self.n_data_pos+1)        #得到一个从1开始的数组，大小为原始数据行数 ex. [1,2,3]
        neg_indices = -1 * np.arange(start=1, stop=self.n_data_neg+1)   #得到一个从-1开始的数组，大小为生成数据行数 ex. [-1,-2,-3,-4]
        self.indicies = np.concatenate([pos_indices, neg_indices])      #链接,ex. [1,2,3,-1,-2,-3,-4]
        # print(pos_indices)                                              #在本例中为[   1    2    3 ... 5344 5345 5346] 长度为原始数据行数
        # print(neg_indices)                                              #在本例中为[-1 -2 …… -500]        长度为生成数据行数
        if self.shuffle:
            random.shuffle(self.indicies)                               #乱序的[-1...-500 1..n_data_pos]
    def on_epoch_end(self):
        self.reset()
        pass

    def __iter__(self):
        return self
