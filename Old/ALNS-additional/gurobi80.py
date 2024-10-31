import pandas as pd
import os
import numpy as np
from gurobipy import *

os.chdir('/Users/apple/Desktop/Study/Traffic/ALNS-additional')

if __name__ == '__main__':
    dis = pd.read_csv('location3.csv', header=None).values
    vehicleNum = 5
    N = 6
    requirement = [31, 16, 25, 44, 61, 56, 58, 62, 8, 71, 29, 24, 69, 23]
    demand = [10, 20, 30, 20, 30, 30, 10, 20, 30, 20, 30, 30]
    e = 3.1415926
    departure = []
    destination = []
    window = [1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 4.836524177878462, 6.477717374902587, 4.346499155632049, 6.030109860947896, 5.314449151747034,6.894066036835748, 1000000]
    start = requirement[0]
    end = requirement[2 * N + 1]
    
    for i in range(1, N + 1):
        departure.append(requirement[i])
    for i in range(1, N + 1):
        destination.append(requirement[N + i])

    model = Model()

    z = model.addVars(2 * N + 2, 2 * N + 2, vehicleNum, vtype=GRB.BINARY, name="z")
    a = model.addVars(2 * N + 2, vtype=GRB.CONTINUOUS, name="a")
    y = model.addVars(2 * N + 2, vtype=GRB.CONTINUOUS, name="y")
    Q = model.addVars(2 * N + 2, vtype=GRB.CONTINUOUS, name="q")
    p = model.addVars(2 * N + 2, vtype=GRB.CONTINUOUS, name="p")  # 增加的惩罚变量
    u = model.addVars(vehicleNum, vtype=GRB.BINARY, name="u")  # 二进制变量，表示是否增加额外成本

    obj = LinExpr(0)
    for i in range(2 * N + 2):
        for j in range(2 * N + 2):
            for k in range(vehicleNum):
                if i != j:
                    obj += z[i, j, k] * dis[i, j]
    for j in range(1, 2 * N + 1):
        obj += e * y[j]
        obj += 2 * e * p[j]  # 加入额外惩罚

    # 计算是否需要额外成本
    for k in range(vehicleNum):
        route_expr = LinExpr(0)
        for i in range(2 * N + 2):
            for j in range(2 * N + 2):
                if i != j:
                    route_expr += z[i, j, k]
        model.addConstr(route_expr <= 1 + u[k] * (2 * N + 1), name=f'u_constr_{k}')  # 如果 u[k] = 0，route_expr <= 1；如果 u[k] = 1，route_expr 可以大于 1
        model.addConstr(route_expr >= 2 * u[k], name=f'u_constr_lower_{k}')  # 如果 route_expr >= 2，u[k] 必须为 1

        obj += u[k] * 2  # 如果u[k] = 1，增加额外成本2

    # Adding constraints with formatted names
    for i in range(1, 2 * N + 1):
        expr1 = LinExpr(0)
        for j in range(2 * N + 2):
            for k in range(vehicleNum):
                if i != j:
                    expr1.addTerms(1, z[i, j, k])
        model.addConstr(expr1 == 1, name=f'cons1_{i}')

    for i in range(1, N + 1):
        for k in range(vehicleNum):
            expr2 = LinExpr(0)
            for j in range(1, 2 * N + 1):
                if i != j:
                    expr2.addTerms(1, z[i, j, k])
                    expr2.addTerms(-1, z[N + i, j, k])
            expr2.addTerms(1, z[i, 2 * N + 1, k])
            expr2.addTerms(-1, z[N + i, 2 * N + 1, k])
            model.addConstr(expr2 == 0, name=f'cons2_{i}_{k}')

    for k in range(vehicleNum):
        expr3 = LinExpr(0)
        for j in range(1, 2 * N + 2):
            expr3.addTerms(1, z[0, j, k])
        model.addConstr(expr3 == 1, name=f'cons3_{k}')

    for i in range(1, 2 * N + 1):
        for k in range(vehicleNum):
            expr4 = LinExpr(0)
            for j in range(2 * N + 2):
                expr4.addTerms(1, z[i, j, k])
                expr4.addTerms(-1, z[j, i, k])
            model.addConstr(expr4 == 0, name=f'cons4_{i}_{k}')

    for k in range(vehicleNum):
        expr5 = LinExpr(0)
        for i in range(2 * N + 1):
            expr5.addTerms(1, z[i, 2 * N + 1, k])
        model.addConstr(expr5 == 1, name=f'cons5_{k}')

    for i in range(2 * N + 2):
        for j in range(2 * N + 2):
            for k in range(vehicleNum):
                if i != j:
                    model.addConstr(a[j] + (1 - z[i, j, k]) * 1000000 >= (a[i] + dis[i, j]) * z[i, j, k], name=f'cons6_{i}_{j}_{k}')
                    model.addConstr(a[j] * z[i, j, k] >= (a[i] + dis[i, j]) * z[i, j, k] - 0.000001, name=f'cons6_{i}_{j}_{k}_epsilon')

    for i in range(1, N + 1):
        model.addConstr(a[N + i] >= a[i] + dis[i, N + i], name=f'cons7_{i}')

    for j in range(1, 2 * N + 1):
        model.addConstr(y[j] >= 0, name=f'cons8_{j}')

    for j in range(1, 2 * N + 1):
        model.addConstr(a[j] >= 0, name=f'cons_a_{j}')

    model.addConstr(a[0] == 0, name='cons_a_0')

    for j in range(N + 1, 2 * N + 1):
        model.addConstr(y[j] >= a[j] - window[j], name=f'constr9_{j}')
        model.addConstr(p[j] >= a[j] - window[j] - 1, name=f'constr10_{j}')
        model.addConstr(p[j] >= 0, name=f'constr101_{j}' )# 超时惩罚条件

    for k in range(vehicleNum):
        for i in range(2 * N + 2):
            for j in range(1, N + 1):
                if i != j:
                    model.addConstr((z[i, j, k] == 1) >> (Q[i] + demand[j - 1] == Q[j]), name=f'cons11_{i}_{j}_{k}')

    for k in range(vehicleNum):
        for i in range(2 * N + 2):
            for j in range(N + 1, 2 * N + 1):
                if i != j:
                    model.addConstr((z[i, j, k] == 1) >> (Q[i] - demand[j - 1] == Q[j]), name=f'cons12_{i}_{j}_{k}')

    for k in range(vehicleNum):
        for j in range(1, N + 1):
            model.addConstr((z[0, j, k] == 1) >> (Q[j] == Q[0] + demand[j - 1]), name=f'cons13_0_{j}_{k}')
            model.addConstr(Q[0] == 0, name=f'cons_q_0_{k}')

    for j in range(2 * N + 2):
        model.addConstr(Q[j] <= 100, name=f'cons_q_max_{j}')

    # Set objective
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    # Check and print results
    if model.status == GRB.OPTIMAL:
        routes = [[] for _ in range(vehicleNum)]
        for k in range(vehicleNum):
            current_route = [(0, a[0].x,Q[0].x,y[0].x,p[0].x)]
            current_node = 0
            while True:
                for j in range(2 * N + 2):
                    if current_node != j and z[current_node, j, k].x > 0.5:
                        current_route.append((j, a[j].x, Q[j].x,y[j].x,p[j].x))
                        current_node = j
                        break
                if current_node == 2 * N + 1:
                    break
            routes[k] = current_route

        for k in range(vehicleNum):
            print(f"Vehicle {k + 1} route:{u[k].x}")

            for node, time, load ,y,p in routes[k]:
                print(f"Node {node} at time {time:.2f} with load {load:.2f} with {y:.2f} and {p:.2f}")

    else:
        print("No optimal solution found")
