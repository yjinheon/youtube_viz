# string matching

import sys

str_list = []

# 
n = int(sys.stdin.readline())

for i in range(n):
    str_list.append(list(map(int,sys.stdin.readLine().split())))

# number of matching strings

n_match = int(sys.stdin.readline())

match_list = []

for i in range(n_match):
    match_list.append(list(map(int,sys.stdin.readline().split())))


# Q1
def str_match(match_list,str_list):
    for i in match_list:
            res = ["YES" if j in i else "NO" for j in str_list]
            print(" ".join(res))


