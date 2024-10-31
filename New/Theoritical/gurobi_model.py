import pandas as pd
import numpy as np
from gurobipy import *

class GurobiModel:
    def __init__(self, demand_file, distance_file):
        self.demand_file = demand_file
        self.distance_file = distance_file
        self.num_vehicles = 5
        self.vehicle_capacity = 100
        self.requirement_position = []
        self.demand = []
        self.time_windows = []
        self.num_customers = 0
        self.model = None

    def load_data(self):
        demand_data = pd.read_csv(self.demand_file, header=0)
        self.requirement_position = demand_data.iloc[:, 0].astype(int).tolist()  # 第一列为坐标
        self.demand = demand_data.iloc[:, 1].astype(int).tolist()  # 第二列为需求
        self.time_windows = demand_data.iloc[:, 2:4].values.tolist()  # 合并为时间窗口
        self.num_customers = len(self.demand) // 2 - 1  # 根据需求数据的长度计算 num_customers

    def build_model(self):
        distance_data = pd.read_csv(self.distance_file, header=None).values
        distance_matrix = np.zeros((2 * self.num_customers + 2, 2 * self.num_customers + 2))
        for i in range(2 * self.num_customers + 2):
            for j in range(2 * self.num_customers + 2):
                distance_matrix[i, j] = distance_data[self.requirement_position[i], self.requirement_position[j]]

        self.model = Model()

        gurobi_distance = self.model.addVars(2 * self.num_customers + 2, 2 * self.num_customers + 2, self.num_vehicles, vtype=GRB.BINARY, name="gurobi_distance")
        gurobi_time = self.model.addVars(2 * self.num_customers + 2, vtype=GRB.CONTINUOUS, name="gurobi_time")
        gurobi_over_time = self.model.addVars(2 * self.num_customers + 2, vtype=GRB.CONTINUOUS, name="gurobi_over_time")
        gurobi_load = self.model.addVars(2 * self.num_customers + 2, vtype=GRB.CONTINUOUS, name="gurobi_load")
        gurobi_penalty = self.model.addVars(2 * self.num_customers + 2, vtype=GRB.CONTINUOUS, name="gurobi_penalty")
        gurobi_vehicle_use_checker = self.model.addVars(self.num_vehicles, vtype=GRB.BINARY, name="gurobi_vehicle_use_checker")


        # 目标函数
        obj = LinExpr(0)
        for i in range(2 * self.num_customers + 2):
            for j in range(2 * self.num_customers + 2):
                for k in range(self.num_vehicles):
                    if i != j:
                        obj += gurobi_distance[i, j, k] * distance_matrix[i, j]
        for j in range(1, 2 * self.num_customers + 1):
            obj += 3.1415926 * gurobi_over_time[j]
            obj += 2 * 3.1415926 * gurobi_penalty[j]

        self.model.setObjective(obj, GRB.MINIMIZE)

        # 约束
        for k in range(self.num_vehicles):
            route_expr = LinExpr(0)
            for i in range(2 * self.num_customers + 2):
                for j in range(2 * self.num_customers + 2):
                    if i != j:
                        route_expr += gurobi_distance[i, j, k]
            self.model.addConstr(route_expr <= 1 + gurobi_vehicle_use_checker[k] * (2 * self.num_customers + 1), name=f'gurobi_vehicle_use_checker_constr_{k}')
            self.model.addConstr(route_expr >= 2 * gurobi_vehicle_use_checker[k], name=f'gurobi_vehicle_use_checker_lower_constr_{k}')

        for i in range(1, 2 * self.num_customers + 1):
            expr1 = LinExpr(0)
            for j in range(2 * self.num_customers + 2):
                for k in range(self.num_vehicles):
                    if i != j:
                        expr1.addTerms(1, gurobi_distance[i, j, k])
            self.model.addConstr(expr1 == 1, name=f'cons1_{i}')

        for i in range(1, self.num_customers + 1):
            for k in range(self.num_vehicles):
                expr2 = LinExpr(0)
                for j in range(1, 2 * self.num_customers + 1):
                    if i != j:
                        expr2.addTerms(1, gurobi_distance[i, j, k])
                        expr2.addTerms(-1, gurobi_distance[self.num_customers + i, j, k])
                expr2.addTerms(1, gurobi_distance[i, 2 * self.num_customers + 1, k])
                expr2.addTerms(-1, gurobi_distance[self.num_customers + i, 2 * self.num_customers + 1, k])
                self.model.addConstr(expr2 == 0, name=f'cons2_{i}_{k}')

        for k in range(self.num_vehicles):
            expr3 = LinExpr(0)
            for j in range(1, 2 * self.num_customers + 2):
                expr3.addTerms(1, gurobi_distance[0, j, k])
            self.model.addConstr(expr3 == 1, name=f'cons3_{k}')

        for i in range(1, 2 * self.num_customers + 1):
            for k in range(self.num_vehicles):
                expr4 = LinExpr(0)
                for j in range(2 * self.num_customers + 2):
                    expr4.addTerms(1, gurobi_distance[i, j, k])
                    expr4.addTerms(-1, gurobi_distance[j, i, k])
                self.model.addConstr(expr4 == 0, name=f'cons4_{i}_{k}')

        for k in range(self.num_vehicles):
            expr5 = LinExpr(0)
            for i in range(2 * self.num_customers + 1):
                expr5.addTerms(1, gurobi_distance[i, 2 * self.num_customers + 1, k])
            self.model.addConstr(expr5 == 1, name=f'cons5_{k}')

        for i in range(2 * self.num_customers + 2):
            for j in range(2 * self.num_customers + 2):
                for k in range(self.num_vehicles):
                    if i != j:
                        self.model.addConstr(gurobi_time[j] + (1 - gurobi_distance[i, j, k]) * 1000000 >= (gurobi_time[i] + distance_matrix[i, j]) * gurobi_distance[i, j, k], name=f'cons6_{i}_{j}_{k}')
                        self.model.addConstr(gurobi_time[j] * gurobi_distance[i, j, k] >= (gurobi_time[i] + distance_matrix[i, j]) * gurobi_distance[i, j, k] - 0.000001, name=f'cons6_{i}_{j}_{k}_epsilon')

        for i in range(1, self.num_customers + 1):
            self.model.addConstr(gurobi_time[self.num_customers + i] >= gurobi_time[i] + distance_matrix[i, self.num_customers + i], name=f'cons7_{i}')
            self.model.addConstr(gurobi_time[i] >= self.time_windows[i - 1][0], name=f'cons_start_time_{i}')
            self.model.addConstr(gurobi_time[i] <= self.time_windows[i - 1][1], name=f'cons_end_time_{i}')

        for j in range(1, 2 * self.num_customers + 1):
            self.model.addConstr(gurobi_over_time[j] >= 0, name=f'cons8_{j}')

        for j in range(1, 2 * self.num_customers + 1):
            self.model.addConstr(gurobi_time[j] >= 0, name=f'cons_a_{j}')

        self.model.addConstr(gurobi_time[0] == 0, name='cons_a_0')

        for j in range(self.num_customers + 1, 2 * self.num_customers + 1):
            self.model.addConstr(gurobi_over_time[j] >= gurobi_time[j] - self.time_windows[j - self.num_customers - 1][1], name=f'constr9_{j}')
            self.model.addConstr(gurobi_penalty[j] >= gurobi_time[j] - self.time_windows[j - self.num_customers - 1][1] - 1, name=f'constr10_{j}')
            self.model.addConstr(gurobi_penalty[j] >= 0, name=f'constr101_{j}')

        for k in range(self.num_vehicles):
            for i in range(2 * self.num_customers + 2):
                for j in range(1, self.num_customers + 1):
                    if i != j:
                        self.model.addConstr((gurobi_distance[i, j, k] == 1) >> (gurobi_load[i] + self.demand[j - 1] == gurobi_load[j]), name=f'cons11_{i}_{j}_{k}')

        for k in range(self.num_vehicles):
            for i in range(2 * self.num_customers + 2):
                for j in range(self.num_customers + 1, 2 * self.num_customers + 1):
                    if i != j:
                        self.model.addConstr((gurobi_distance[i, j, k] == 1) >> (gurobi_load[i] - self.demand[j - 1] == gurobi_load[j]), name=f'cons12_{i}_{j}_{k}')

        for k in range(self.num_vehicles):
            for j in range(1, self.num_customers + 1):
                self.model.addConstr((gurobi_distance[0, j, k] == 1) >> (gurobi_load[j] == gurobi_load[0] + self.demand[j - 1]), name=f'cons13_0_{j}_{k}')
                self.model.addConstr(gurobi_load[0] == 0, name=f'cons_q_0_{k}')

        for j in range(2 * self.num_customers + 2):
            self.model.addConstr(gurobi_load[j] <= self.vehicle_capacity, name=f'cons_q_max_{j}')

    def optimize(self):
        self.model.optimize()

    def output_results(self):
        if self.model.status == GRB.OPTIMAL:
            routes = [[] for _ in range(self.num_vehicles)]
            for k in range(self.num_vehicles):
                current_route = [(0, self.model.getVarByName("gurobi_time[0]").x, self.model.getVarByName("gurobi_load[0]").x, self.model.getVarByName("gurobi_over_time[0]").x, self.model.getVarByName("gurobi_penalty[0]").x)]
                current_node = 0
                while True:
                    for j in range(2 * self.num_customers + 2):
                        if current_node != j and self.model.getVarByName(f"gurobi_distance[{current_node},{j},{k}]").x > 0.5:
                            current_route.append((j, self.model.getVarByName(f"gurobi_time[{j}]").x, self.model.getVarByName(f"gurobi_load[{j}]").x, self.model.getVarByName(f"gurobi_over_time[{j}]").x, self.model.getVarByName(f"gurobi_penalty[{j}]").x))
                            current_node = j
                            break
                    if current_node == 2 * self.num_customers + 1:
                        break
                routes[k] = current_route

            for k in range(self.num_vehicles):
                print(f"Vehicle {k + 1} route: {self.model.getVarByName(f'gurobi_vehicle_use_checker[{k}]').x}")
                for node, time, load, overtime, penalty in routes[k]:
                    print(f"Node {node} at time {time:.2f} with load {load:.2f} with {overtime:.2f} and {penalty:.2f}")
        else:
            print("No optimal solution found.")

# 使用示例
if __name__ == "__main__":
    model = GurobiModel('New/Data/demand_data.csv', 'New/Data/wangjing-newdis-simplify.csv')
    model.load_data()
    model.build_model()
    model.optimize()
    model.output_results()