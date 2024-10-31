import pandas as pd
import numpy as np
from gurobipy import *

class GurobiModel:
    def __init__(self, demand_file, distance_file):
        self.demand_file = demand_file
        self.distance_file = distance_file
        self.vehicleNum = 5
        self.max_capacity = 100
        self.requirementPosition = []
        self.demand = []
        self.start_time = []
        self.end_time = []
        self.requirementNum = 0
        self.model = None

    def load_data(self):
        # 读取需求数据
        demand_data = pd.read_csv(self.demand_file, header=0)
        self.requirementPosition = demand_data.iloc[:, 0].astype(int).tolist()  # 第一列为坐标
        self.demand = demand_data.iloc[:, 1].astype(int).tolist()  # 第二列为需求
        self.start_time = demand_data.iloc[:, 2].tolist()  # 第三列为开始时间
        self.end_time = demand_data.iloc[:, 3].tolist()  # 第四列为结束时间
        self.requirementNum = len(self.demand) // 2 - 1  # 根据需求数据的长度计算 requirementNum

    def build_model(self):
        distanceData = pd.read_csv(self.distance_file, header=None).values
        # 创建 dis 矩阵
        dis = np.zeros((2 * self.requirementNum + 2, 2 * self.requirementNum + 2))
        for i in range(2 * self.requirementNum + 2):
            for j in range(2 * self.requirementNum + 2):
                dis[i, j] = distanceData[self.requirementPosition[i], self.requirementPosition[j]]

        self.model = Model()

        gurobiDistance = self.model.addVars(2 * self.requirementNum + 2, 2 * self.requirementNum + 2, self.vehicleNum, vtype=GRB.BINARY, name="gurobiDistance")
        gurobiTime = self.model.addVars(2 * self.requirementNum + 2, vtype=GRB.CONTINUOUS, name="gurobiTime")
        gurobiOverTime = self.model.addVars(2 * self.requirementNum + 2, vtype=GRB.CONTINUOUS, name="gurobiOverTime")
        gurobiDemand = self.model.addVars(2 * self.requirementNum + 2, vtype=GRB.CONTINUOUS, name="gurobiDemand")
        gurobiPenalty = self.model.addVars(2 * self.requirementNum + 2, vtype=GRB.CONTINUOUS, name="gurobiPenalty")
        gurobiVehicleUseChecker = self.model.addVars(self.vehicleNum, vtype=GRB.BINARY, name="gurobiVehicleUseChecker")

        # 目标函数和约束的构建
        self.set_objective_and_constraints(dis, gurobiDistance, gurobiTime, gurobiOverTime, gurobiDemand, gurobiPenalty, gurobiVehicleUseChecker)

    def set_objective_and_constraints(self, dis, gurobiDistance, gurobiTime, gurobiOverTime, gurobiDemand, gurobiPenalty, gurobiVehicleUseChecker):
        obj = LinExpr(0)
        for i in range(2 * self.requirementNum + 2):
            for j in range(2 * self.requirementNum + 2):
                for k in range(self.vehicleNum):
                    if i != j:
                        obj += gurobiDistance[i, j, k] * dis[i, j]
        for j in range(1, 2 * self.requirementNum + 1):
            obj += 3.1415926 * gurobiOverTime[j]
            obj += 2 * 3.1415926 * gurobiPenalty[j]

        for k in range(self.vehicleNum):
            route_expr = LinExpr(0)
            for i in range(2 * self.requirementNum + 2):
                for j in range(2 * self.requirementNum + 2):
                    if i != j:
                        route_expr += gurobiDistance[i, j, k]
            self.model.addConstr(route_expr <= 1 + gurobiVehicleUseChecker[k] * (2 * self.requirementNum + 1), name=f'gurobiVehicleUseChecker_constr_{k}')
            self.model.addConstr(route_expr >= 2 * gurobiVehicleUseChecker[k], name=f'gurobiVehicleUseChecker_lower_constr_{k}')
            obj += gurobiVehicleUseChecker[k] * 2

        # 其他约束的添加
        # ... (根据您的原始代码添加其他约束)

        self.model.setObjective(obj, GRB.MINIMIZE)

    def optimize(self):
        self.model.optimize()

    def output_results(self):
        if self.model.status == GRB.OPTIMAL:
            routes = [[] for _ in range(self.vehicleNum)]
            for k in range(self.vehicleNum):
                current_route = [(0, self.model.getVarByName("gurobiTime[0]").x, self.model.getVarByName("gurobiDemand[0]").x, self.model.getVarByName("gurobiOverTime[0]").x, self.model.getVarByName("gurobiPenalty[0]").x)]
                current_node = 0
                while True:
                    for j in range(2 * self.requirementNum + 2):
                        if current_node != j and self.model.getVarByName(f"gurobiDistance[{current_node},{j},{k}]").x > 0.5:
                            current_route.append((j, self.model.getVarByName(f"gurobiTime[{j}]").x, self.model.getVarByName(f"gurobiDemand[{j}]").x, self.model.getVarByName(f"gurobiOverTime[{j}]").x, self.model.getVarByName(f"gurobiPenalty[{j}]").x))
                            current_node = j
                            break
                    if current_node == 2 * self.requirementNum + 1:
                        break
                routes[k] = current_route

            for k in range(self.vehicleNum):
                print(f"Vehicle {k + 1} route: {self.model.getVarByName(f'gurobiVehicleUseChecker[{k}]').x}")
                for node, time, load, overtime, penalty in routes[k]:
                    print(f"Node {node} at time {time:.2f} with load {load:.2f} with {overtime:.2f} and {penalty:.2f}")
        else:
            print("No optimal solution found")

# 使用示例
if __name__ == "__main__":
    model = GurobiModel('New/Data/demand_data.csv', 'New/Data/wangjing-newdis-simplify.csv')
    model.load_data()
    model.build_model()
    model.optimize()
    model.output_results()