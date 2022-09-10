import math
import cx_Oracle
import os

def rule_sample(path_rules,path, order):

    #rule_test = 'rule_test.txt'
    conn = cx_Oracle.connect('system', 'Pjfpjf11', '127.0.0.1:1521/orcl')  # 连接数据库
    cursor = conn.cursor()
    sql1 = "select * from \"" + path + "\" "
    print(sql1)
    cursor.execute(sql1)
    data_all = cursor.fetchall()  # 全体数据
    data_all = [x[:-1] for x in data_all]  # 去掉Label列

    if order == 1:
        print("正序采样数据以生产规则……")

    elif order == 0:
        print("逆序采样数据以生产规则……")
        data_all = [x[::-1] for x in data_all]

    rule = ''
    for data_tuple in data_all:
        rule_tuple = ''
        for data_cell in data_tuple:
            # print(data_cell)
            # print(type(data_cell))
            if data_cell == None:
                continue
            else:
                if rule_tuple == '':
                    rule_tuple = data_cell
                else:
                    rule_tuple += ',' + (data_cell)
        rule += rule_tuple + '\n'
        # print(type(rule))
    top=os.getcwd()
    path_rules = os.path.join( 'data', 'save', path_rules)
    # print(path_rules)
    # print(len(data_all))
    with open(path_rules, 'w', encoding='utf-8') as f:
        f.write(rule)
    return len(data_all)
    # print(data1)



if __name__ == '__main__':
    order = 0
    path = 'Hosp_rules'
    path_rules = 'rules.txt'
    rule_sample(path_rules,path, order)
    # dict_generator()
