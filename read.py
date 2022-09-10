import cx_Oracle

def read_data():
    conn = cx_Oracle.connect('system', 'Pjfpjf11', '127.0.0.1:1521/orcl')  # 连接数据库
    cursor = conn.cursor()
    cursor.execute('select * from "Hosp_rules_copy" ')#"City","State" ,where rownum<=10
    rows = cursor.fetchall()
    t = ''
    with open('data/save/read_test.txt', 'w') as f:
        for i in rows:
            for e in range(len(rows[0])):
                t = t + str(i[e]) + ' '
            f.write(t.strip(', '))
            f.write('\n')
            t = ''
    f.close()
    cursor.close()
    conn.close()


if __name__ == '__main__':
    read_data()
