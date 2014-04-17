# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

def dfs(start, end, path, t):
    path = path + [start]
    if start == end:
        if t == 0:
            return path
        else:
            return []
    elif t < 0 :
        return []
    else:
        for w in range(N + 2):
            if T[start][w] >= 0: # avoid cycle
                ans = dfs(w, end, path, t - T[start][w])
                if ans != []:
                    return ans
        path = path[:-1]
        return []

def connected(start, end):
    visited[start] = 1
    if start == end:
        return True
    else:
        for w in range(N + 2):
            if T[start][w] >= 0:
                if visited[w] == 0:
                    if connected(w, end):
                        return True
        return False

# <codecell>

N, Ne, Nw, F = map(int, raw_input().split())
CityE = map(int, raw_input().split())
Pi = map(int, raw_input().split())
P = sum(Pi)
CityW = map(int, raw_input().split())
T = [[-1 for x in range(N + 2)] for j in range(N + 2)]
C = [[0 for x in range(N + 2)] for j in range(N + 2)]
for i in range(F):
    [a,b,c,d] = map(int, raw_input().split())
    T[a][b] = d
    C[a][b] = c
for i in range(Ne): # build super source
    T[N][CityE[i]] = 0
    C[N][CityE[i]] = Pi[i]
for j in range(Nw): # build super tank
    T[CityW[j]][N + 1] = 0
    C[CityW[j]][N + 1] = P

# check connected
t = 0
flow = 0
dummy = True
for i in range(Ne):
    visited = [0 for x in range(N + 2)]
    if connected(CityE[i], N + 1) == False:
        print(-1)
        dummy = False
        break
if dummy == True:
    while P > 0:
        while True:
            path = dfs(N, N + 1, [], t)
            if path == []:
                t += 1
                break
            else:
                # print "t",t,"path",path
                min_flow = float("inf")
                for i in range(1,len(path)):
                    temp = C[path[i - 1]][path[i]]
                    if temp < min_flow:
                        min_flow = temp
                flow += min_flow
                for i in range(1,len(path) - 1):
                    C[path[i]][path[i - 1]] += min_flow
                    # if i > 1:
                    #     T[path[i]][path[i - 1]] = T[path[i - 1]][path[i]]
                    C[path[i - 1]][path[i]] -= min_flow
                    if C[path[i - 1]][path[i]] <= 0:  
                        T[path[i - 1]][path[i]] = -1
        # print "t", t - 1, "flow", flow
        P -= flow
    print t - 1

# <codecell>


