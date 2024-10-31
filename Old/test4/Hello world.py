import numpy as np
import pandas as pd
import random
import math
import os

os.chdir('/Users/apple/Desktop/Study/Traffic/test4')

class VRPPD:
    def __init__(self, num_customers, num_vehicles, demands, vehicle_capacity, distance_matrix, time_windows, penalty_coefficient):
        self.num_customers = num_customers
        self.num_vehicles = num_vehicles
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        self.distance_matrix = distance_matrix
        self.time_windows = time_windows
        self.penalty_coefficient = penalty_coefficient
        self.routes = self.initial_solution()
        print("初始解生成完成")

    def initial_solution(self):
        routes = [[] for _ in range(self.num_vehicles)]
        remaining_customers = list(range(1, self.num_customers + 1))
        vehicle_load = [0] * self.num_vehicles
        current_time = [0] * self.num_vehicles  # Track current time for each vehicle

        # Start each vehicle from depot (index 0)
        current_locations = [0] * self.num_vehicles
        '''
        print("开始生成初始解")
        print(f"剩余客户：{remaining_customers}")
        print(f"车辆负载：{vehicle_load}")
        print(f"当前时间：{current_time}")
        print(f"当前地点：{current_locations}")'''

        while remaining_customers:
            for vehicle in range(self.num_vehicles):
                if remaining_customers:
                    best_customer = None
                    best_cost = float('inf')

                    for customer in remaining_customers:
                        cost = self.distance_matrix[current_locations[vehicle]][customer]
                        if cost < best_cost and vehicle_load[vehicle] + self.demands[customer - 1] <= self.vehicle_capacity:
                            best_cost = cost
                            best_customer = customer

                    if best_customer is not None:
                        routes[vehicle].append(best_customer)
                        vehicle_load[vehicle] += self.demands[best_customer - 1]
                        current_locations[vehicle] = best_customer
                        remaining_customers.remove(best_customer)

                        '''print(f"车辆 {vehicle} 选择客户 {best_customer}")
                        print(f"更新后的剩余客户：{remaining_customers}")
                        print(f"更新后的车辆负载：{vehicle_load}")
                        print(f"更新后的当前地点：{current_locations}")'''
        
        # Ensure each route ends with depot
        for i in range(self.num_vehicles):
            if routes[i] and routes[i][-1] != 0:
                routes[i].append(0)

        '''print("初始解生成完成：", routes)'''

        return routes

    def total_distance(self, routes):
        total_distance = 0
        for route in routes:
            if len(route) > 0:
                total_distance += self.distance_matrix[0][route[0]]
                for i in range(1, len(route)):
                    total_distance += self.distance_matrix[route[i - 1]][route[i]]
                total_distance += self.distance_matrix[route[-1]][[2*self.num_customers + 1]]
        return total_distance

    def total_penalty(self, routes):
        total_penalty = 0
        for route in routes:
            current_time = 0
            for i in range(len(route)):
                if i == 0:
                    current_time += self.distance_matrix[0][route[i]]
                else:
                    current_time += self.distance_matrix[route[i - 1]][route[i]]
                if route[i] >= len(self.time_windows):
                    raise IndexError(f"Route index {route[i]} out of range for time windows")
                if current_time > self.time_windows[route[i]][1]:
                    total_penalty += (current_time - self.time_windows[route[i]][1]) * self.penalty_coefficient
                if current_time < self.time_windows[route[i]][0]:
                    current_time = self.time_windows[route[i]][0]
        return total_penalty

    def neighbor(self, routes):
        new_routes = [route[:] for route in routes]
        route1, route2 = random.sample(range(len(routes)), 2)
        if len(new_routes[route1]) > 0 and len(new_routes[route2]) > 0:
            customer1 = random.choice(new_routes[route1])
            customer2 = random.choice(new_routes[route2])
            idx1 = new_routes[route1].index(customer1)
            idx2 = new_routes[route2].index(customer2)
            new_routes[route1][idx1], new_routes[route2][idx2] = new_routes[route2][idx2], new_routes[route1][idx1]
            # Check vehicle capacity after swap
            load1 = sum(self.demands[customer - 1] for customer in new_routes[route1])
            load2 = sum(self.demands[customer - 1] for customer in new_routes[route2])
        
            if load1 > self.vehicle_capacity or load2 > self.vehicle_capacity:
                # Revert the swap if capacity constraint is violated
                new_routes[route1][idx1], new_routes[route2][idx2] = new_routes[route2][idx2], new_routes[route1][idx1]
        return new_routes

    def simulated_annealing(self, initial_temp, cooling_rate, stopping_temp):
        current_temp = initial_temp
        current_solution = self.routes
        current_distance = self.total_distance(current_solution)
        current_penalty = self.total_penalty(current_solution)
        current_cost = current_distance + current_penalty
        best_solution = current_solution
        best_cost = current_cost

        while current_temp > stopping_temp:
            new_solution = self.neighbor(current_solution)
            new_distance = self.total_distance(new_solution)
            new_penalty = self.total_penalty(new_solution)
            new_cost = new_distance + new_penalty

            if new_cost < current_cost or random.uniform(0, 1) < np.exp((current_cost - new_cost) / current_temp):
                current_solution = new_solution
                current_cost = new_cost

                if current_cost < best_cost:
                    best_solution = current_solution
                    best_cost = current_cost

            current_temp *= cooling_rate

        return best_solution, best_cost

def print_solution(solution):
    for i, route in enumerate(solution):
        print(f"Vehicle {i + 1} route:")
        print(route)
        print()

def main():
    # 读取数据
    df_test = pd.read_csv('test.csv')
    demands = df_test['demand'].tolist()
    time_windows = df_test[['start_time', 'end_time']].values.tolist()
    
    distance_matrix = pd.read_csv('location.csv', header=None).values

    num_customers = len(demands) // 2  # Adjusted for pickup and delivery
    num_vehicles = int(input("请输入车辆数量: "))
    vehicle_capacity = int(input("请输入每辆车的容量: "))
    penalty_coefficient = 3.1415926
    
    vrppd = VRPPD(num_customers, num_vehicles, demands, vehicle_capacity, distance_matrix, time_windows, penalty_coefficient)
    print("贪心算法初始解：")
    print_solution(vrppd.routes)
    print("初始解总成本：", vrppd.total_distance(vrppd.routes) + vrppd.total_penalty(vrppd.routes))
    
    # 使用模拟退火算法优化初始解
    initial_temp = 10000
    cooling_rate = 0.999
    stopping_temp = 0.001

    best_solution, best_cost = vrppd.simulated_annealing(initial_temp, cooling_rate, stopping_temp)
    print("\n模拟退火算法优化后的解：")
    print_solution(best_solution)
    print("最小成本：", best_cost)

if __name__ == "__main__":
    main()
