import cx_Oracle

def reset(path_ori,path):

    # path_ori = "LETTER"
    # path = "LETTER_copy"
    # path_ori = "Hosp_rules"
    # path = "Hosp_rules_copy"
    # path_ori = "Test"
    # path = "Test_copy"
    print("重置数据集",path)
    conn = cx_Oracle.connect('system', 'Pjfpjf11', '127.0.0.1:1521/orcl')  # 连接数据库
    cursor = conn.cursor()
    # sql= 'INSERT INTO "Hosp_rules_copy1" SELECT * FROM "Hosp_rules" ORDER BY "Provider ID"'
    # sql ='DELETE FROM "Hosp2_rule_copy" '
    sql = "DELETE FROM \"" + path + "\" "
    cursor.execute(sql)
    conn.commit()   #清空

    sql1 = "select * from \"" + path_ori + "\" "    #where rownum < 3  #order by "Provider ID" desc
    # print(sql1)
    cursor.execute(sql1)
    data1 = cursor.fetchall()
    # print(data1)
    # data1.reverse()
    # print(type(data1))
    t2 = len(data1[0])   # 每行数据长度，-1是为了变成索引位置

    #print("1")
    # print(data1)
    for row in data1:
        # print(row)
        for num in range(t2):
            if num == 0:
                sql_before = "'%s'"
            else:
                sql_before = sql_before + ",'%s'"
        # print(sql_before)

        va = []
        for num in range(t2):
            va.append(row[num])
        sql_after = tuple(va)
        # print(sql_after)


        sql3 = "insert into \"" + path + "\" values(" + sql_before + ")"
        sql4=sql3% (sql_after)
        # print(sql4)
        cursor.execute(sql4)
        conn.commit()   #重置

    cursor.close()
    conn.close()
    print("重置完成")

def dict_generator():
    path="Hosp_rules_copy"
    conn = cx_Oracle.connect('system', 'Pjfpjf11', '127.0.0.1:1521/orcl')  # 连接数据库
    cursor = conn.cursor()
    sql1 = "select * from \"" + path + "\" "
    print(sql1)
    cursor.execute(sql1)
    data1 = cursor.fetchall()  # 全体数据
    des = cursor.description
    t2 = len(data1[0]) - 1  # 每行数据长度，-1是为了变成索引位置

    att_name = []
    for item in des:
        att_name.append(item[0])
    # print(att_name)
    dict = {}
    for i in range(t2):  # -1是为了去掉label列
        dict[i] = att_name[i]
    print(dict)
    f = open('data/save/att_name.txt', 'w')
    f.write(str(dict))
    f.close()
    cursor.close()
    conn.close()

if __name__ == '__main__':

    flag = 2
    if flag == 1:
        path_ori = "Test"  # Hosp_rules
        path = "Test_copy"  #
    if flag == 2:
        path_ori = "Hosp_rules"  #
        path = "Hosp_rules_copy"

    reset(path_ori,path)
    # dict_generator()

