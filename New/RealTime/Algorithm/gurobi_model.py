import pandas as pd

import numpy as np
from gurobipy import *

# 定义全局变量
global vehicleNum, gurobiTime, gurobiDemand, gurobiOverTime, gurobiPenalty, requirementNum, gurobiVehicleUseChecker, gurobiDistance

def Gurobi():
    global vehicleNum, gurobiTime, gurobiDemand, gurobiOverTime, gurobiPenalty, requirementNum, gurobiVehicleUseChecker, gurobiDistance
    distanceData = pd.read_csv('New/Data/wangjing-newdis-simplify.csv', header=None).values
    vehicleNum = 5
    requirementPosition = [31, 16, 25, 44, 61, 56, 58, 62, 8, 71, 29, 24, 69, 23]
    penalty_coefficient = 3.1415926
    demand_data = pd.read_csv('New/Data/demand_data.csv', header=0).values
    requirementPosition = demand_data[:, 0].tolist()  # 第一列为坐标
    demand = demand_data[:, 1].tolist()  # 第二列为需求
    start_time = demand_data[:, 2].tolist()  # 第三列为开始时间
    end_time = demand_data[:, 3].tolist()  # 第四列为结束时间
    requirementNum = len(demand) // 2 - 1  # 根据需求数据的长度计算 requirementNum
    

    # 创建 dis 矩阵
    dis = np.zeros((2 * requirementNum + 2, 2 * requirementNum + 2))
    for i in range(2 * requirementNum + 2):
        for j in range(2 * requirementNum + 2):
            dis[i, j] = distanceData[requirementPosition[i], requirementPosition[j]]

    model = Model()

    gurobiDistance = model.addVars(2 * requirementNum + 2, 2 * requirementNum + 2, vehicleNum, vtype=GRB.BINARY, name="gurobiDistance")
    gurobiTime = model.addVars(2 * requirementNum + 2, vtype=GRB.CONTINUOUS, name="gurobiTime")
    gurobiOverTime = model.addVars(2 * requirementNum + 2, vtype=GRB.CONTINUOUS, name="gurobiOverTime")
    gurobiDemand = model.addVars(2 * requirementNum + 2, vtype=GRB.CONTINUOUS, name="gurobiDemand")
    gurobiPenalty = model.addVars(2 * requirementNum + 2, vtype=GRB.CONTINUOUS, name="gurobiPenalty")
    gurobiVehicleUseChecker = model.addVars(vehicleNum, vtype=GRB.BINARY, name="gurobiVehicleUseChecker")

    obj = LinExpr(0)
    for i in range(2 * requirementNum + 2):
        for j in range(2 * requirementNum + 2):
            for k in range(vehicleNum):
                if i != j:
                    obj += gurobiDistance[i, j, k] * dis[i, j]
    for j in range(1, 2 * requirementNum + 1):
        obj += penalty_coefficient * gurobiOverTime[j]
        obj += 2 * penalty_coefficient * gurobiPenalty[j]

    for k in range(vehicleNum):
        route_expr = LinExpr(0)
        for i in range(2 * requirementNum + 2):
            for j in range(2 * requirementNum + 2):
                if i != j:
                    route_expr += gurobiDistance[i, j, k]
        model.addConstr(route_expr <= 1 + gurobiVehicleUseChecker[k] * (2 * requirementNum + 1), name=f'gurobiVehicleUseChecker_constr_{k}')
        model.addConstr(route_expr >= 2 * gurobiVehicleUseChecker[k], name=f'gurobiVehicleUseChecker_lower_constr_{k}')

        obj += gurobiVehicleUseChecker[k] * 2

    for i in range(1, 2 * requirementNum + 1):
        expr1 = LinExpr(0)
        for j in range(2 * requirementNum + 2):
            for k in range(vehicleNum):
                if i != j:
                    expr1.addTerms(1, gurobiDistance[i, j, k])
        model.addConstr(expr1 == 1, name=f'cons1_{i}')

    for i in range(1, requirementNum + 1):
        for k in range(vehicleNum):
            expr2 = LinExpr(0)
            for j in range(1, 2 * requirementNum + 1):
                if i != j:
                    expr2.addTerms(1, gurobiDistance[i, j, k])
                    expr2.addTerms(-1, gurobiDistance[requirementNum + i, j, k])
            expr2.addTerms(1, gurobiDistance[i, 2 * requirementNum + 1, k])
            expr2.addTerms(-1, gurobiDistance[requirementNum + i, 2 * requirementNum + 1, k])
            model.addConstr(expr2 == 0, name=f'cons2_{i}_{k}')

    for k in range(vehicleNum):
        expr3 = LinExpr(0)
        for j in range(1, 2 * requirementNum + 2):
            expr3.addTerms(1, gurobiDistance[0, j, k])
        model.addConstr(expr3 == 1, name=f'cons3_{k}')

    for i in range(1, 2 * requirementNum + 1):
        for k in range(vehicleNum):
            expr4 = LinExpr(0)
            for j in range(2 * requirementNum + 2):
                expr4.addTerms(1, gurobiDistance[i, j, k])
                expr4.addTerms(-1, gurobiDistance[j, i, k])
            model.addConstr(expr4 == 0, name=f'cons4_{i}_{k}')

    for k in range(vehicleNum):
        expr5 = LinExpr(0)
        for i in range(2 * requirementNum + 1):
            expr5.addTerms(1, gurobiDistance[i, 2 * requirementNum + 1, k])
        model.addConstr(expr5 == 1, name=f'cons5_{k}')

    for i in range(2 * requirementNum + 2):
        for j in range(2 * requirementNum + 2):
            for k in range(vehicleNum):
                if i != j:
                    model.addConstr(gurobiTime[j] + (1 - gurobiDistance[i, j, k]) * 1000000 >= (gurobiTime[i] + dis[i, j]) * gurobiDistance[i, j, k], name=f'cons6_{i}_{j}_{k}')
                    model.addConstr(gurobiTime[j] * gurobiDistance[i, j, k] >= (gurobiTime[i] + dis[i, j]) * gurobiDistance[i, j, k] - 0.000001, name=f'cons6_{i}_{j}_{k}_epsilon')

    for i in range(1, requirementNum + 1):
        model.addConstr(gurobiTime[requirementNum + i] >= gurobiTime[i] + dis[i, requirementNum + i], name=f'cons7_{i}')

    for j in range(1, 2 * requirementNum + 1):
        model.addConstr(gurobiOverTime[j] >= 0, name=f'cons8_{j}')

    for j in range(1, 2 * requirementNum + 1):
        model.addConstr(gurobiTime[j] >= 0, name=f'cons_a_{j}')

    model.addConstr(gurobiTime[0] == 0, name='cons_a_0')

    for j in range(requirementNum + 1, 2 * requirementNum + 1):
        model.addConstr(gurobiOverTime[j] >= gurobiTime[j] - end_time[j - requirementNum - 1], name=f'constr9_{j}')
        model.addConstr(gurobiPenalty[j] >= gurobiTime[j] - end_time[j - requirementNum - 1] - 1, name=f'constr10_{j}')
        model.addConstr(gurobiPenalty[j] >= 0, name=f'constr101_{j}')

    for k in range(vehicleNum):
        for i in range(2 * requirementNum + 2):
            for j in range(1, requirementNum + 1):
                if i != j:
                    model.addConstr((gurobiDistance[i, j, k] == 1) >> (gurobiDemand[i] + demand[j - 1] == gurobiDemand[j]), name=f'cons11_{i}_{j}_{k}')

    for k in range(vehicleNum):
        for i in range(2 * requirementNum + 2):
            for j in range(requirementNum + 1, 2 * requirementNum + 1):
                if i != j:
                    model.addConstr((gurobiDistance[i, j, k] == 1) >> (gurobiDemand[i] - demand[j - 1] == gurobiDemand[j]), name=f'cons12_{i}_{j}_{k}')

    for k in range(vehicleNum):
        for j in range(1, requirementNum + 1):
            model.addConstr((gurobiDistance[0, j, k] == 1) >> (gurobiDemand[j] == gurobiDemand[0] + demand[j - 1]), name=f'cons13_0_{j}_{k}')
            model.addConstr(gurobiDemand[0] == 0, name=f'cons_q_0_{k}')

    for j in range(2 * requirementNum + 2):
        model.addConstr(gurobiDemand[j] <= 100, name=f'cons_q_max_{j}')

    # 在约束中添加对gurobiTime的限制
    for i in range(1, requirementNum + 1):
        model.addConstr(gurobiTime[i] >= start_time[i - 1], name=f'cons_start_time_{i}')

    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    return model


def output_results(model):
    global vehicleNum, gurobiTime, gurobiDemand, gurobiOverTime, gurobiPenalty, requirementNum, gurobiVehicleUseChecker, gurobiDistance
    if model.status == GRB.OPTIMAL:
        routes = [[] for _ in range(vehicleNum)]
        for k in range(vehicleNum):
            current_route = [(0, gurobiTime[0].x, gurobiDemand[0].x, gurobiOverTime[0].x, gurobiPenalty[0].x)]
            current_node = 0
            while True:
                for j in range(2 * requirementNum + 2):
                    if current_node != j and gurobiDistance[current_node, j, k].x > 0.5:
                        current_route.append((j, gurobiTime[j].x, gurobiDemand[j].x, gurobiOverTime[j].x, gurobiPenalty[j].x))
                        current_node = j
                        break
                if current_node == 2 * requirementNum + 1:
                    break
            routes[k] = current_route

        for k in range(vehicleNum):
            print(f"Vehicle {k + 1} route: {gurobiVehicleUseChecker[k].x}")
            for node, time, load, overtime, penalty in routes[k]:
                print(f"Node {node} at time {time:.2f} with load {load:.2f} with {overtime:.2f} and {penalty:.2f}")

    else:
        print("No optimal solution found")


def print_optimal_results(model):
    global gurobiTime
    if model.status == GRB.OPTIMAL:
        print("Optimal solution found:")
        print(f"Objective value: {model.objVal}")  # 输出目标值
    else:
        print("No optimal solution found.")


# 调用建模和输出结果
model = Gurobi()
output_results(model)
print_optimal_results(model)