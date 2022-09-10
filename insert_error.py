import cx_Oracle
import random
import reset


def insert_error(path_ori,path,error_rate):

    conn = cx_Oracle.connect('system', 'Pjfpjf11', '127.0.0.1:1521/orcl')  # 连接数据库
    cursor = conn.cursor()

    random.seed(1)
    path2 = path + "1"
    path3 = path + "2"
    reset.reset(path_ori,path)   #首先，重置
    # reset.reset(path_ori, path2)

    sql = "DELETE FROM \"" + path2 + "\" "
    cursor.execute(sql)
    conn.commit()  # 清空

    sql = "DELETE FROM \"" + path3 + "\" "
    cursor.execute(sql)
    conn.commit()  # 清空

    sql1 = "select * from \"" + path + "\" "
    print(sql1)
    cursor.execute(sql1)
    data1 = cursor.fetchall()#全体数据
    des = cursor.description
    # print("表的描述:", des)
    # print("表头:", ",".join([item[0] for item in des]))
    t1 = len(data1)   # 总数据量          #-1可变为索引位置
    t2 = len(data1[0]) - 1  # 每行数据长度，-1是为了去掉label列

    att_name=[]
    for item in des:
        att_name.append(item[0])
    # print(att_name)
    dict={}
    for i in range(t2):    #
        dict[i]=att_name[i]
    print(dict)
    # f = open('att_name.txt', 'w')
    f = open('data/save/att_name.txt', 'w')
    f.write(str(dict))
    f.close()

    # f = open('data/save/att_name.txt', 'r')
    # label2att = eval(f.read())
    # f.close()
    # print(label2att)

    att = list(dict.values())

    count_error_all=int(error_rate*t1)
    # print(l2)

    # count_error_MV= int(count_error_all/3)      #missing value
    # count_error_FI= int(count_error_all/3)      #formatting issue
    # count_error_typo=count_error_all-count_error_MV-count_error_FI


    error_index=random.sample(range(0,t1),count_error_all)    #0-9999共10000个数中随机10个不同的数
    print(len(error_index))
    # print(error_index)
    count=0
    for index in error_index:
        row=data1[index]
        # print(row)

        for i in range(t2):  # t2
            if i == 0:
                sql_inf = "\"" + att[i] + "\"='" + row[i] + "'"
            else:
                sql_inf = sql_inf + " and \"" + att[i] + "\"='" + row[i] + "'"
        sql_info = sql_inf + " and \"Label\"='None'"
        # print(sql_info)
        if count<int(count_error_all/3):
            error_list = ["error1", "error2", "error3", "error4", "error5"]
            r = random.randint(1, t2 - 1)  # 假设一共9列，计算t2时-1除掉最后一列label，现在再-1是为了变成索引位置，所以在1-7中随机数
            r2 = random.randint(0, 4)
            error = error_list[r2]
        elif count<int(2*count_error_all/3):
            r = random.randint(1, t2 - 1)
            error="missing"                 #注意, 这里可写error=" "作为缺失值, 但执行模型检测时将所有空值置换为"missing",因此这里直接写成missing
        else:
            r = random.randint(1, t2 - 1)
            sql = "select distinct (\"" + att[r] + "\") from \"" + path + "\""
            # print(sql)
            cursor.execute(sql)
            values = cursor.fetchall()
            error_value = random.choice(values) #同一列的其他值
            error=error_value[0]
            while (error==row[r]):
                error_value = random.choice(values)  # 同一列的其他值
                error = error_value[0]
                # print(att[r], error, row[r])
        # print(r)
        # print(error)
        if (error is None):
            error=""
        sql2 = "update \"" + path + "\" set \"Label\"='1' , \"" + att[r] + "\"='" + error + "' where  " + sql_info + "" #and \"" + att[r] + "\"='error'
        # print(sql2)
        cursor.execute(sql2)
        conn.commit()



        # 生成Hosp_rules_copy2
        sql = "select * from \"" + path_ori + "\" where  " + sql_inf + ""
        cursor.execute(sql)
        data_clean = cursor.fetchall()
        # print(sql)
        # print(data_clean)
        t3 = len(data_clean[0])  # 每行数据长度
        for num in range(t3):
            if num == 0:
                sql_before = "'%s'"
            else:
                sql_before = sql_before + ",'%s'"
        # print(sql_before)
        va = []
        for num in range(t3):
            va.append(data_clean[0][num])
        sql_after = tuple(va)
        sql_clean = "insert into \"" + path3 + "\" values(" + sql_before + ")"
        sql3=sql_clean% (sql_after)
        # print(sql4)
        cursor.execute(sql3)
        conn.commit()   #重置

        # 生成Hosp_rules_copy1
        sql4 = "insert into \"" + path2 + "\" values(" + sql_before + ")"
        sql5 = sql4 % (sql_after)
        # print(sql4)
        cursor.execute(sql5)
        conn.commit()
        sql_dirty = "update \"" + path2 + "\" set \"Label\"='1' , \"" + att[
            r] + "\"='" + error + "' where  " + sql_inf + ""
        cursor.execute(sql_dirty)
        conn.commit()
        count = count + 1



    sql_check = "select * from \"" + path + "\" where \"Label\"='1'"
    cursor.execute(sql_check)
    data2 = cursor.fetchall()
    print("生成错误数量:", len(data2))    #生成错误多余预期可能是由于数据集中存在重复项

    cursor.close()
    conn.close()


if __name__ == '__main__':
    path_ori="Hosp_rules"
    path = "Hosp_rules_copy"
    # path_ori = "Food"
    # path = "Food_copy"
    # path_ori = "Gov_data"
    # path = "Gov_data_copy"
    # path_ori = "Flight"
    # path = "Flight_copy"

    error_rate = 0.1
    insert_error(path_ori,path,error_rate)
    print("插入错误完成")


