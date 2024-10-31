import numpy as np
import random
import math
import pandas as pd
import os
os.chdir('/Users/apple/Desktop/Study/Traffic/ALNS-additional')

# 初始化蚁群算法的参数
class ACO_VRPPD:
    def __init__(self, num_ants, num_iterations, alpha, beta, rho, Q, num_customers, vehicle_capacity, vehicle_num, distance_matrix, demands):
        self.num_ants = num_ants            # 蚂蚁数量
        self.num_iterations = num_iterations  # 迭代次数
        self.alpha = alpha                  # 信息素权重
        self.beta = beta                    # 启发式信息权重
        self.rho = rho                      # 信息素挥发系数
        self.Q = Q                          # 信息素增量系数
        self.num_customers = num_customers  # 客户数量
        self.vehicle_capacity = vehicle_capacity  # 车辆容量
        self.vehicle_num = vehicle_num      # 车辆数量
        self.distance_matrix = distance_matrix    # 距离矩阵
        self.demands = demands              # 每个客户的需求
        self.num_nodes = 2 * num_customers + 2  # 包括仓库和接送点

        # 初始化信息素矩阵
        self.pheromone_matrix = np.ones((self.num_nodes, self.num_nodes)) / self.num_nodes

        # 初始化启发式信息矩阵（这里使用距离的倒数作为启发式信息）
        self.heuristic_matrix = 1 / (self.distance_matrix + np.eye(self.num_nodes))

    def run(self):
        best_solution = None
        best_cost = float('inf')

        for iteration in range(self.num_iterations):
            ants = [Ant(self) for _ in range(self.num_ants)]  # 初始化蚂蚁

            for ant in ants:
                ant.construct_solution()  # 每只蚂蚁构建一个解

            # 更新信息素
            self.update_pheromone(ants)

            # 找到最优解
            for ant in ants:
                if ant.total_cost < best_cost:
                    best_cost = ant.total_cost
                    best_solution = ant.solution

            print(f"Iteration {iteration + 1}: Best Cost = {best_cost}")

        return best_solution, best_cost

    # 更新信息素矩阵
    def update_pheromone(self, ants):
        self.pheromone_matrix *= (1 - self.rho)  # 信息素挥发

        for ant in ants:
            for vehicle_route in ant.solution:
                for i in range(len(vehicle_route) - 1):
                    from_node = vehicle_route[i]
                    to_node = vehicle_route[i + 1]
                    self.pheromone_matrix[from_node, to_node] += self.Q / ant.total_cost

class Ant:
    def __init__(self, aco):
        self.aco = aco
        self.solution = [[] for _ in range(self.aco.vehicle_num)]  # 每辆车的路径
        self.visited = set()
        self.vehicle_loads = [0] * self.aco.vehicle_num  # 每辆车的负载
        self.current_nodes = [0] * self.aco.vehicle_num  # 每辆车的当前位置
        self.total_cost = 0

    def construct_solution(self):
        # 初始化每辆车的剩余容量和当前位置
        self.vehicle_loads = [0] * self.aco.vehicle_num
        self.current_nodes = [0] * self.aco.vehicle_num
        self.solution = [[0] for _ in range(self.aco.vehicle_num)]  # 每辆车从仓库出发

        while len(self.visited) < self.aco.num_customers:
            for vehicle in range(self.aco.vehicle_num):
                next_node = self.select_next_node(vehicle)  # 为每辆车选择下一个节点
                if next_node is None:
                    continue

                self.move_to_next_node(vehicle, next_node)  # 将该车辆移动到下一个节点

            if all(len(route) == 1 for route in self.solution):  # 如果所有车都无法继续，则退出
                break

        # 每辆车回到仓库
        for vehicle in range(self.aco.vehicle_num):
            self.total_cost += self.aco.distance_matrix[self.current_nodes[vehicle], 2 * self.aco.num_customers + 1]
            self.solution[vehicle].append(2 * self.aco.num_customers + 1)

    def select_next_node(self, vehicle):
        candidates = []
        for customer in range(1, self.aco.num_customers + 1):
            if customer not in self.visited:
                if self.vehicle_loads[vehicle] + self.aco.demands[customer - 1] <= self.aco.vehicle_capacity:
                    candidates.append(customer)

        if not candidates:
            return None

        # 计算每个候选节点的选择概率
        probabilities = []
        for customer in candidates:
            pheromone = self.aco.pheromone_matrix[self.current_nodes[vehicle], customer]
            heuristic = self.aco.heuristic_matrix[self.current_nodes[vehicle], customer]
            probabilities.append((pheromone ** self.aco.alpha) * (heuristic ** self.aco.beta))

        probabilities = probabilities / np.sum(probabilities)
        return np.random.choice(candidates, p=probabilities)

    def move_to_next_node(self, vehicle, next_node):
        self.solution[vehicle].append(next_node)
        self.visited.add(next_node)

        # 更新总成本
        self.total_cost += self.aco.distance_matrix[self.current_nodes[vehicle], next_node]

        # 更新车辆的负载
        self.vehicle_loads[vehicle] += self.aco.demands[next_node - 1]
        self.current_nodes[vehicle] = next_node

if __name__ == '__main__':
    # 从test.csv读取数据
    df_test = pd.read_csv('test5.csv')
    demands = df_test['demand'].tolist()  # 从test.csv读取demand列
    num_customers = len(demands) // 2  # 客户数（pickup和delivery数量相等）
    
    # 从location3.csv读取距离矩阵
    distance_matrix = pd.read_csv('location3.csv', header=None).values

    # 询问用户输入车辆容量和车辆数量
    vehicle_capacity = int(input("请输入每辆车的容量: "))
    vehicle_num = int(input("请输入车辆数量: "))

    # 初始化蚁群优化算法的参数
    aco_vrppd = ACO_VRPPD(
        num_ants=10,
        num_iterations=100,
        alpha=1.0,
        beta=5.0,
        rho=0.5,
        Q=100,
        num_customers=num_customers,
        vehicle_capacity=vehicle_capacity,
        vehicle_num=vehicle_num,
        distance_matrix=distance_matrix,
        demands=demands
    )

    # 运行蚁群算法
    best_solution, best_cost = aco_vrppd.run()

    print("最佳路径：", best_solution)
    print("最佳成本：", best_cost)
