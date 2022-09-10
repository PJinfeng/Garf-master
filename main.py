import numpy as np
from SeqGAN.train import Trainer
from SeqGAN.get_config import get_config
import cx_Oracle
import time
from reset import reset
from insert_error import insert_error
from eva import evaluate
from att_reverse import att_reverse
from rule_sample import rule_sample

config = get_config('config.ini')


path = config["path_pos"]
path_ori = path.strip('_copy')
print(path_ori)
# path_ori = "UIS2"  #Hosp_reverse
# path = "UIS2_copy" #Hosp_reverse
# path_ori = "UIS"  #
# path = "UIS_copy"
f = open('data/save/log_evaluation.txt', 'w')
f.write("")
f.close()
error_rate = 0.1

# reset(path_ori, path)
insert_error(path_ori, path, error_rate)    #第一遍运行时开，反向运行时关

starttime = time.time()
print("加载config，定义各项变量")

flag = 2  # 0代表训练SeqGAN，1代表修复部分,2代表同时进行
order = 1   # 顺序，1代表正序，0代表逆序

att_reverse(path,order)
if (flag == 0 or flag == 2):
    trainer = Trainer(order,
                      config["batch_size"],
                      config["max_length"],
                      config["g_e"],
                      config["g_h"],
                      config["d_e"],
                      config["d_h"],
                      config["d_dropout"],
                      config["generate_samples"],
                      path_pos=config["path_pos"],
                      path_neg=config["path_neg"],
                      g_lr=config["g_lr"],
                      d_lr=config["d_lr"],
                      n_sample=config["n_sample"],
                      path_rules=config["path_rules"])
    # Pretraining for adversarial training 对抗训练的pretrain,运行这部分时要batch_size = 32
    print("开始预训练")
    # insert_error.insert_error(1000)
    trainer.pre_train(g_epochs=config["g_pre_epochs"],  # 50
                      d_epochs=config["d_pre_epochs"],  # 1
                      g_pre_path=config["g_pre_weights_path"],  # data/save/generator_pre.hdf5
                      d_pre_path=config["d_pre_weights_path"],  # data/save/discriminator_pre.hdf5
                      g_lr=config["g_pre_lr"],  # 1e-2
                      d_lr=config["d_pre_lr"])  # 1e-4

    trainer.load_pre_train(config["g_pre_weights_path"], config["d_pre_weights_path"])
    trainer.reflect_pre_train()  # 将layer层权重映射给agent

    trainer.train(steps=1,  # 一共就1个大回合，每个大回合中生成器训练1次，判别器训练1次
                  g_steps=1,
                  head=10,
                  g_weights_path=config["g_weights_path"],
                  d_weights_path=config["d_weights_path"])

    trainer.save(config["g_weights_path"], config["d_weights_path"])
if (flag == 1 or flag == 2):
    trainer = Trainer(order,
                      1,
                      config["max_length"],
                      config["g_e"],
                      config["g_h"],
                      config["d_e"],
                      config["d_h"],
                      config["d_dropout"],
                      config["generate_samples"],
                      path_pos=config["path_pos"],
                      path_neg=config["path_neg"],
                      g_lr=config["g_lr"],
                      d_lr=config["d_lr"],
                      n_sample=config["n_sample"],
                      path_rules=config["path_rules"])
    trainer.load(config["g_weights_path"], config["d_weights_path"])

    rule_len=rule_sample(config["path_rules"],path,order)
    # trainer.generate_rules(config["path_rules"], config["generate_samples"])  # 生产rule sequence，即rules.txt

    trainer.train_rules(rule_len,config["path_rules"])  # 用于生产规则,通过rules.txt生成rules_final.txt

    trainer.filter(config["path_pos"])

    f = open('data/save/log.txt', 'w')
    f.write("")
    f.close()
    att_reverse(path, 1)
    trainer.repair(config["path_pos"])  # 用于双向修复，规则来自于rules_final.txt,修复的数据是Hosp_rules_copy，是Hosp_rules的完全备份+随机噪声


    # # trainer.repair_SeqGAN()

endtime = time.time()
dtime = endtime - starttime
print("本次修复运行时长为", dtime, "错误率为", error_rate)

evaluate(path_ori, path)
    # error_rate = error_rate + 10


# while (error_rate < 0.9):


#
# trainer.test()
#
# trainer.generate_txt(config["g_test_path"], config["generate_samples"])
# trainer.predict_rules() #已经没用了，仅作为测试
