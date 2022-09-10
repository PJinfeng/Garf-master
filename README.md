# Garf-master
# Garf-master

实验需要在数据库中存在4个数据集，以Hosp_rules为例(config.ini中path_pos = Hosp_rules_copy，若变更数据集则变更此处)  
Hosp_rules是初始干净数据集  
Hosp_rules_copy是添加了错误数据的版本，也是我们目标的待修复数据集，初始为空，通过insert_error.py复制蓝本并添加错误  
Hosp_rules_copy1是添加错误数据时单独拿出的错误数据，初始为空  
Hosp_rules_copy2是添加的错误数据所对应的真实正确数据，初始为空  

注意：  
数据集Hosp_rules不参与检测修复过程，仅作为数据蓝本，用于结果评估，相关代码仅生效于insert_error.py，reset.py和eva.py  
数据集Hosp_rules_copy1和Hosp_rules_copy2则仅在生成错误时产生，为了方便对照观察，与程序无关，若不需要则在insert_error.py中删除“path2”和“path3”相关部分即可  
自己添加的数据集需要在最后一列添加Label列，但全置空即可，仅在eva.py进行结果评估时使用，但由于代码过程中包含去除Label列影响，缺少会影响结果或报错  

本代码已进行模块化拆分，默认单向训练并保存模型结果，实际使用时请至少正向运行一次，反向运行一次，多次运行能少量提升性能结果  
在main.py中order = 1代表正向；order = 0代表反向，第二次运行时请勿重新添加错误数据，请注释掉insert_error(path_ori, path, error_rate)  
insert_error.py用于添加错误，错误包含3类：拼写错误，数据缺失，同属性列下其他值随机替换，若不需要则同样注释掉insert_error(path_ori, path, error_rate)  
提高config.ini中g_pre_epochs，d_pre_epochs数值(即模型生成器和判别器迭代次数)可少量提升性能，但时间代价增大  

预期结果：  
测试数据集数据量为10k条，Hosp数据集结果为准确率98%±1%，召回率65%±3%；Food数据集结果为准确率97%±2%，召回率62%±5%  
随着数据量提升，模型性能提升，论文中Hosp数据量为100k，Food数据量为200k  
若需补充数据，请点击下方链接：  

Hosp：http://www.hospitalcompare.hhs.gov/或https://data.medicare.gov/data/physician-compare  
Food：https://data.cityofchicago.org  
Flight：http://lunadong.com/fusionDataSets.htm  
UIS：https://www.cs.utexas.edu/users/ml/riddle/data.html  




