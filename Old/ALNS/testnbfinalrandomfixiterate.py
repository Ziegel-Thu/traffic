import numpy as np
import pandas as pd
import random
import math
import os
import copy
import itertools
import sys

os.chdir('/Users/apple/Desktop/Study/Traffic/Test5')

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
        print("初始解：", self.routes)

    def initial_solution(self):
        routes = [[] for _ in range(self.num_vehicles)]
        remaining_customers = list(range(1, self.num_customers + 1))
        remaining_deliveries = list(range(self.num_customers + 1, 2 * self.num_customers + 1))
        vehicle_load = [0] * self.num_vehicles
        current_time = [0] * self.num_vehicles  # Track current time for each vehicle

        # Start each vehicle from depot (index 0)
        current_locations = [0] * self.num_vehicles

        while remaining_customers or any(load > 0 for load in vehicle_load):
            for vehicle in range(self.num_vehicles):
                if not remaining_customers and vehicle_load[vehicle] == 0:
                    continue

                best_customer = None
                best_delivery = None
                best_cost = float('inf')

                # Check for the nearest pickup
                for customer in remaining_customers:
                    cost = self.distance_matrix[current_locations[vehicle]][customer]
                    if cost < best_cost and vehicle_load[vehicle] + self.demands[customer - 1] <= self.vehicle_capacity:
                        best_cost = cost
                        best_customer = customer
                        best_delivery = None

                # Check for the nearest delivery
                for delivery in remaining_deliveries:
                    if (delivery - self.num_customers) in routes[vehicle]:
                        cost = self.distance_matrix[current_locations[vehicle]][delivery]
                        if cost < best_cost:
                            best_cost = cost
                            best_customer = None
                            best_delivery = delivery

                if best_customer is not None:
                    routes[vehicle].append(best_customer)
                    vehicle_load[vehicle] += self.demands[best_customer - 1]
                    current_locations[vehicle] = best_customer
                    remaining_customers.remove(best_customer)

                if best_delivery is not None:
                    routes[vehicle].append(best_delivery)
                    vehicle_load[vehicle] -= self.demands[best_delivery - self.num_customers - 1]
                    current_locations[vehicle] = best_delivery
                    remaining_deliveries.remove(best_delivery)

        # Ensure each route ends with the final depot (index 2*num_customers + 1)
        final_depot = 2 * self.num_customers + 1
        for i in range(self.num_vehicles):
            if routes[i][-1] != final_depot:
                routes[i].append(final_depot)

        print("生成的初始路径：", routes)
        return routes

    def total_distance(self, routes):
        total_distance = 0
        final_depot = 2 * self.num_customers + 1
        for route in routes:
            if len(route) > 0:
                total_distance += self.distance_matrix[0][route[0]]
                for i in range(1, len(route)):
                    total_distance += self.distance_matrix[route[i - 1]][route[i]]
                total_distance += self.distance_matrix[route[-1]][final_depot]
        return total_distance

    def total_penalty(self, routes):
        total_penalty = 0
        final_depot = 2 * self.num_customers + 1
        for route in routes:
            current_time = 0
            for i in range(len(route)):
                if i == 0:
                    current_time += self.distance_matrix[0][route[i]]
                else:
                    current_time += self.distance_matrix[route[i - 1]][route[i]]

                # Skip time window checks for depot and final depot
                if route[i] == 0 or route[i] == final_depot:
                    continue

                if current_time > self.time_windows[route[i]-1][1]:
                    total_penalty += (current_time - self.time_windows[route[i]-1][1]) * self.penalty_coefficient
                if current_time < self.time_windows[route[i]-1][0]:
                    current_time = self.time_windows[route[i]-1][0]
        return total_penalty

    def calculate_distance_penalty(self, route):
        total_distance = 0
        total_penalty = 0
        current_time = 0
        final_depot = 2 * self.num_customers + 1

        for i in range(len(route)):
            if i == 0:
                total_distance += self.distance_matrix[0][route[i]]
                current_time += self.distance_matrix[0][route[i]]
            else:
                total_distance += self.distance_matrix[route[i - 1]][route[i]]
                current_time += self.distance_matrix[route[i - 1]][route[i]]

            if route[i] == 0 or route[i] == final_depot:
                continue

            if current_time > self.time_windows[route[i]-1][1]:
                total_penalty += (current_time - self.time_windows[route[i]-1][1]) * self.penalty_coefficient
            if current_time < self.time_windows[route[i]-1][0]:
                current_time = self.time_windows[route[i]-1][0]

        total_distance += self.distance_matrix[route[-1]][final_depot]
        return total_distance, total_penalty

def print_solution(solution, vrppd):
    total_distance = 0
    total_penalty = 0

    for i, route in enumerate(solution):
        distance, penalty = vrppd.calculate_distance_penalty(route)
        total_distance += distance
        total_penalty += penalty

        # Verify route distance using distance matrix
        calculated_distance = 0
        for j in range(len(route) - 1):
            calculated_distance += vrppd.distance_matrix[route[j]][route[j+1]]

        # Add distance from depot to first customer and last customer to depot
        calculated_distance += vrppd.distance_matrix[0][route[0]]
        calculated_distance += vrppd.distance_matrix[route[-1]][2 * vrppd.num_customers + 1]

        print(f"Vehicle {i + 1} route: {route}")
        print(f"  Distance: {distance} (Calculated: {calculated_distance})")
        print(f"  Penalty: {penalty}")
        print()

    print(f"Total Distance: {total_distance}")
    print(f"Total Penalty: {total_penalty}")

def ALNS(vrppd, max_iterations=100000, destruction_level=0.3, initial_temperature=100.0, cooling_rate=0.99):
    current_solution = vrppd.routes
    best_solution = current_solution
    best_cost = vrppd.total_distance(current_solution) + vrppd.total_penalty(current_solution)
    all_time_best_solution = current_solution
    all_time_best_cost = best_cost
    temperature = initial_temperature

    recent_solutions = []
    max_recent_solutions = 50
    identical_solution_threshold = 20

    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        print(f"Iteration {iteration}")
        if recent_solutions.count(best_cost) >= identical_solution_threshold:
            print("Detected local optimum. Performing random destroy and repair.")

            # Perform random destroy and repair
            destroyed_customers, new_solution = random_destroy(best_solution, vrppd, destruction_level)
            current_solution, new_cost, flags = random_repair(new_solution, destroyed_customers, vrppd)
            while flags == 1:
                print("随机失败，继续随机")
                destroyed_customers, new_solution = random_destroy(best_solution, vrppd, destruction_level)
                current_solution, new_cost, flags = random_repair(new_solution, destroyed_customers, vrppd)

            # Reset the temperature
            temperature = initial_temperature
            print("Temperature reset.")

        # Perform worst destroy
        destroyed_customers, new_solution = worst_destroy(current_solution, vrppd, destruction_level)
        print(f"第 {iteration} 次迭代后的新解（破坏后）：{new_solution}")

        # Perform repair
        repaired_solution, new_cost, flag = repair(new_solution, destroyed_customers, vrppd)
        if flag == 0:
            repaired_solution, new_cost = current_solution, new_cost
            print(current_solution)
            print("修复失败，进入随机模式")
            destroyed_customers, new_solution = random_destroy(best_solution, vrppd, destruction_level)
            current_solution, new_cost, flags = random_repair(new_solution, destroyed_customers, vrppd)
            while flags == 1:
                print("随机失败，继续随机")
                destroyed_customers, new_solution = random_destroy(best_solution, vrppd, destruction_level)
                current_solution, new_cost, flags = random_repair(new_solution, destroyed_customers, vrppd)
        else:
            print(f"第 {iteration} 次迭代后的当前解（修复后）：{repaired_solution}")
        
        kix = 0
        for routes in repaired_solution:
            kix += len(routes)
        if kix != 23:
            print('end')
            sys.exit()
        print(f"Current Solution Cost: {new_cost}")

        # Simulated annealing acceptance criterion
        cost_diff = new_cost - best_cost
        if cost_diff < 0 or random.random() < math.exp(-cost_diff / temperature):
            current_solution = repaired_solution
            best_cost = new_cost
            best_solution = current_solution
            print(f"New solution accepted with cost: {new_cost}")
        else:
            print(f"New solution rejected with cost: {new_cost}")

        # Update all-time best solution if the current solution is better
        if best_cost < all_time_best_cost:
            all_time_best_cost = best_cost
            all_time_best_solution = best_solution
            print(f"New all-time best solution found with cost: {all_time_best_cost}")

        # Add the current solution to the recent solutions list
        recent_solutions.append(new_cost)
        if len(recent_solutions) > max_recent_solutions:
            recent_solutions.pop(0)

        # Reduce the temperature
        temperature *= cooling_rate

    print("\nFinal Solution:")
    print_solution(all_time_best_solution, vrppd)
    print("All-Time Best Solution Cost:", all_time_best_cost)

def worst_destroy(solution, vrppd, destruction_level):
    remove_customers = []
    new_solution = copy.deepcopy(solution)
    distance_matrix = vrppd.distance_matrix
    destroy_num = math.ceil(destruction_level * vrppd.num_customers)

    while len(remove_customers) < destroy_num:
        best_fitness = float("inf")
        best_customer = None
        best_route = None

        route = random.choice(new_solution)
        for i in range(0, len(route) - 1):
            pickup = route[i]
            if pickup <= vrppd.num_customers:  # Ensure it's a pickup node
                delivery = pickup + vrppd.num_customers
                if delivery in route:
                    pickup_index = route.index(pickup)
                    delivery_index = route.index(delivery)

                    if len(route) <= 3:
                        fitness = vrppd.total_distance([route])
                    else:
                        node0 = route[pickup_index - 1]
                        node1 = route[delivery_index + 1]
                        fitness = distance_matrix[node0][node1] - distance_matrix[node0][pickup] - distance_matrix[pickup][delivery] - distance_matrix[delivery][node1]

                    if fitness < best_fitness:
                        best_fitness = fitness
                        best_customer = pickup
                        best_route = route

        if best_customer is not None:
            pickup = best_customer
            delivery = pickup + vrppd.num_customers
            best_route.remove(pickup)
            best_route.remove(delivery)
            remove_customers.append(pickup)
            if len(best_route) > 2 and best_route[-1] != 2 * vrppd.num_customers + 1:  # Ensure the route ends at the depot
                best_route.append(2 * vrppd.num_customers + 1)

    print("破坏后移除的客户：", remove_customers)
    return remove_customers, new_solution

def random_destroy(solution, vrppd, destruction_level):
    remove_customers = []
    new_solution = copy.deepcopy(solution)
    destroy_num = math.ceil(destruction_level * vrppd.num_customers)

    while len(remove_customers) < destroy_num:
        route = random.choice(new_solution)
        if len(route) > 2:  # 确保不是只包含仓库的路径
            pickup = random.choice(route[:-1])  # 随机选择一个客户（不是仓库）
            if pickup <= vrppd.num_customers:
                delivery = pickup + vrppd.num_customers
                if delivery in route:
                    route.remove(pickup)
                    route.remove(delivery)
                    remove_customers.append(pickup)
                    if len(route) > 2 and route[-1] != 2 * vrppd.num_customers + 1:
                        route.append(2 * vrppd.num_customers + 1)  # Return to depot
    print("随机破坏后移除的客户：", remove_customers)
    return remove_customers, new_solution

def repair(solution, destroyed_customers, vrppd):
    num_vehicles = vrppd.num_vehicles
    best_solution = None
    best_total_cost = float('inf')
    flag = 0

    # 生成所有可能的客户分配组合
    all_combinations = list(itertools.product(range(num_vehicles), repeat=len(destroyed_customers)))

    for combination in all_combinations:
        # 初始化一个新的解决方案作为当前组合的尝试
        new_solution = copy.deepcopy(solution)
        valid = True

        # 逐个将客户分配到各自分配的车辆路线中
        for i, vehicle in enumerate(combination):
            customer = destroyed_customers[i]
            delivery = customer + vrppd.num_customers

            # 检查插入客户和交付点后，是否满足容量约束
            route_load = sum(vrppd.demands[node - 1] for node in new_solution[vehicle] if node <= vrppd.num_customers)
            if route_load + vrppd.demands[customer - 1] > vrppd.vehicle_capacity:
                valid = False
                break

            # 遍历所有可能的插入位置
            best_position = None
            best_position_cost = float('inf')

            for insert_pos in range(len(new_solution[vehicle]) ):
                for delivery_pos in range(insert_pos, len(new_solution[vehicle]) ):
                    temp_route = new_solution[vehicle][:insert_pos] + [customer] + new_solution[vehicle][insert_pos:delivery_pos] + [delivery] + new_solution[vehicle][delivery_pos:]
                    
                    # 计算插入后的总成本
                    temp_solution = copy.deepcopy(new_solution)
                    temp_solution[vehicle] = temp_route
                    current_total_distance = vrppd.total_distance(temp_solution)
                    current_total_penalty = vrppd.total_penalty(temp_solution)
                    current_total_cost = current_total_distance + current_total_penalty
                    
                    # 如果当前插入位置成本更低，则更新最佳插入位置
                    if current_total_cost < best_position_cost:
                        best_position_cost = current_total_cost
                        best_position = (insert_pos, delivery_pos)

            # 如果找到最佳插入位置，将客户插入到该位置
            if best_position:
                insert_pos, delivery_pos = best_position
                new_solution[vehicle] = new_solution[vehicle][:insert_pos] + [customer] + new_solution[vehicle][insert_pos:delivery_pos] + [delivery] + new_solution[vehicle][delivery_pos:]

        # 计算整个解决方案的总成本
        if valid:
            final_total_distance = vrppd.total_distance(new_solution)
            final_total_penalty = vrppd.total_penalty(new_solution)
            final_total_cost = final_total_distance + final_total_penalty

            # 如果当前方案的总成本更低，则更新最佳方案
            if final_total_cost < best_total_cost:
                best_total_cost = final_total_cost
                best_solution = new_solution
                flag = 1

    # 如果找到有效的组合，返回最佳解和对应的成本，否则返回当前解和一个错误标志
    if flag:
        return best_solution, best_total_cost, flag
    else:
        return solution, float('inf'), 0

def random_repair(solution, destroyed_customers, vrppd):
    new_solution = copy.deepcopy(solution)
    flag = 0
    for customer in destroyed_customers:
        count = 1
        while count <= 20 and count > 0:
            delivery = customer + vrppd.num_customers

            # 随机选择一条路线
            route = random.choice(new_solution)

            # 选择插入位置，确保不会插入到终点（回到仓库）之后
            max_index = len(route) - 1  # 终点的前一个位置
            i = random.randint(0, max_index)
            j = random.randint(i, max_index)

            # 创建新的路径
            new_route = route[:i] + [customer] + route[i:j] + [delivery] + route[j:]

            # Check if the insertion satisfies the vehicle capacity constraint
            route_load = sum(vrppd.demands[node - 1] for node in new_route if node <= vrppd.num_customers)
            if route_load <= vrppd.vehicle_capacity:
                new_solution[new_solution.index(route)] = new_route
                count = 0
            else:
                count += 1
            if count > 20:
                flag = 1

    return new_solution, vrppd.total_distance(new_solution) + vrppd.total_penalty(new_solution), flag

def main():
    # 读取数据
    df_test = pd.read_csv('test.csv')
    demands = df_test['demand'].tolist()
    time_windows = df_test[['start_time', 'end_time']].values.tolist()

    distance_matrix = pd.read_csv('location2.csv', header=None).values

    num_customers = len(demands) // 2  # Adjusted for pickup and delivery
    num_vehicles = int(input("请输入车辆数量: "))
    vehicle_capacity = int(input("请输入每辆车的容量: "))
    penalty_coefficient = 3.1415926

    vrppd = VRPPD(num_customers, num_vehicles, demands, vehicle_capacity, distance_matrix, time_windows, penalty_coefficient)
    print("贪心算法初始解：")
    print_solution(vrppd.routes, vrppd)
    initial_routes = vrppd.routes
    print("初始解总成本：", vrppd.total_distance(vrppd.routes) + vrppd.total_penalty(vrppd.routes))

    # 运行ALNS算法
    ALNS(vrppd)
    print(initial_routes)

if __name__ == "__main__":
    main()