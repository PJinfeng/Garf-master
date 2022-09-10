import cx_Oracle

def att_reverse(path,order):


    conn = cx_Oracle.connect('system', 'Pjfpjf11', '127.0.0.1:1521/orcl')  # 连接数据库
    cursor = conn.cursor()
    sql1 = "select * from \"" + path + "\" "
    print(sql1)
    cursor.execute(sql1)
    data1 = cursor.fetchall()  # 全体数据
    data1 = [x[:-1] for x in data1]     #数据去掉label
    if order == 0:
        data1 = [x[::-1] for x in data1]    #逆序排列
        des = cursor.description
        des.reverse()
        print(des)
        del (des[0])    #表头去掉label
    else:
        des = cursor.description
    print(data1)

    # print(type(des))
    # print("表的描述:", des)
    # print("表头:", ",".join([item[0] for item in des]))
    t1 = len(data1)  # 总数据量
    t2 = len(data1[0])  # 每行数据长度

    att_name = []
    for item in des:
        # print(item)
        att_name.append(item[0])
    # print(att_name)
    dict = {}
    for i in range(t2):
        dict[i] = att_name[i]
    print(dict)
    # f = open('att_name.txt', 'w')
    f = open('data/save/att_name.txt', 'w')
    f.write(str(dict))
    f.close()


if __name__ == '__main__':


    path = "Hosp_rules_copy"

    att_reverse(path,1)
    # dict_generator()

