import numpy as np
import cx_Oracle


def addtwodimdict(thedict, key_a, key_b, val):
  if key_a in thedict:
    thedict[key_a].update({key_b: val})
  else:
    thedict.update({key_a:{key_b: val}})

users = {
    str(['1','2']):{
    'first':'yu',
    'last':'lei',
    'location':'hs',
    },
    'B':{
    'first':'liu',
    'last':'wei',
    'location':'hs',
    },
    'E':{
        'reason':{'aa':'bb'}
    }
}
users[str(['1','2'])]['first']='zz'
# print(a)
users['C']=['xx','yy']
users['B'].update({'first': 'xu'})
# users.update({'D':{'xx':'yy','ww':'zz'}})
# a=users
# b=users
#
# # a.update({'D':{'aa':'bb'}})
# b['D'].update({'aa':'bb'})
# print(a)
# print(b)

print(users)
addtwodimdict(users,'D','xx','yy')
print(users)
addtwodimdict(users,'D','xx','yy')
print(users)
addtwodimdict(users,'D','xx','zz')
print(users)
addtwodimdict(users,'D','yy','zz')
print(users)
addtwodimdict(users['E'],'reason','cc','dd')
print(users)
addtwodimdict(users,'F','aa','dd')
addtwodimdict(users,'F','reason',{"1":"2"})
addtwodimdict(users,'F','reason',{"3":"4"})
print(users)

del users['E']['reason']['aa']
print(users)
addtwodimdict(users['E'],'reason','aa','bb')
print(users)
# word_=['1','2']
# word_=[]
# l=len(word_)
# for i in range(l):
#     print(word_[i])



# f = open('data/save/rules_final.txt', 'r')
# rules_final = eval(f.read())
# f.close()
# print(rules_final)
# for rulename,ruleinfo in rules_final.items():
#     print("rulename:"+rulename)
#     print("ruleinfo:",ruleinfo)
#     # print("___________")
#     print("reason:", ruleinfo['reason'])
#     print("result:", ruleinfo['result'])
#     print("trust:", ruleinfo['trust'])
#     k=list(ruleinfo['reason'].keys())
#     reason=k[0]
#     v = list(ruleinfo['reason'].values())
#     reason_ = v[0]
#     k = list(ruleinfo['result'].keys())
#     result = k[0]
#     v = list(ruleinfo['result'].values())
#     result_ = v[0]
#
#     trust = ruleinfo['trust']
#     # print(reason)
#     # print(result)
#     # print(trust)
#     # print()
#     # for k,v in items():
#     #     print(k)




# # username,userinfo 相当于 key,value
# for username,userinfo in users.items():
#     print("username:"+username)
#     print("userinfo"+str(userinfo))
#     fullname=userinfo['first']+" "+userinfo['last']
#     print("fullname:"+fullname)
#     print("location:"+userinfo['location'])