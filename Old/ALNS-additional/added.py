import pandas as pd
import numpy as np
import pandas as pd
import random
import math
import os
import copy
import sys

os.chdir('/Users/apple/Desktop/Study/Traffic/ALNS-additional')

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

    def calculate_route_distance(self, route):
        total_distance = 0
        final_depot = 2 * self.num_customers + 1

        if len(route) > 0:
            total_distance += self.distance_matrix[0][route[0]]
            for i in range(1, len(route)):
                total_distance += self.distance_matrix[route[i - 1]][route[i]]
            total_distance += self.distance_matrix[route[-1]][final_depot]
        return total_distance

    def calculate_route_penalty(self, route):
        total_penalty = 0
        current_time = 0
        final_depot = 2 * self.num_customers + 1
        
        if len(route )!=1:
            total_penalty+=2
        for i in range(len(route)):
            if i == 0:
                current_time += self.distance_matrix[0][route[i]]
            else:
                current_time += self.distance_matrix[route[i - 1]][route[i]]

            if route[i] == 0 or route[i] == final_depot:
                continue

            if current_time > self.time_windows[route[i]-1][1]:
                total_penalty += (current_time - self.time_windows[route[i]-1][1]) * self.penalty_coefficient
            if current_time > self.time_windows[route[i]-1][1]+1:
                total_penalty += (current_time- self.time_windows[route [i]-1][1]-1)* self.penalty_coefficient*2
            if current_time < self.time_windows[route[i]-1][0]:
                current_time = self.time_windows[route[i]-1][0]
            #print(total_penalty,route[i],current_time)


        return total_penalty

    def calculate_total_cost(self, routes):
        total_distance = sum(self.calculate_route_distance(route) for route in routes)
        total_penalty = sum(self.calculate_route_penalty(route) for route in routes)
        return total_distance + total_penalty

def print_solution(solution, vrppd):
    total_distance = 0
    total_penalty = 0

    for i, route in enumerate(solution):
        distance = vrppd.calculate_route_distance(route)
        penalty = vrppd.calculate_route_penalty(route)
        total_distance += distance
        total_penalty += penalty

        print(f"Vehicle {i + 1} route: {route}")
        print(f"  Distance: {distance}")
        print(f"  Penalty: {penalty}")
        print()

    print(f"Total Distance: {total_distance}")
    print(f"Total Penalty: {total_penalty}")
    print(f"Total Cost: {total_distance + total_penalty}")

def ALNS(vrppd, max_iterations=10000, destruction_level=0.3, initial_temperature=100.0, cooling_rate=0.99):
    current_solution = vrppd.routes
    best_solution = current_solution
    best_cost = vrppd.calculate_total_cost(current_solution)
    all_time_best_solution = current_solution
    all_time_best_cost = best_cost
    temperature = initial_temperature

    recent_solutions = []
    max_recent_solutions = 10
    identical_solution_threshold = 4

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
        if kix != 2*vrppd.num_customers+vrppd.num_vehicles:
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
                        fitness = vrppd.calculate_total_cost([route])
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
    best_solution = copy.deepcopy(solution)
    best_total_cost = float('inf')
    flag = 1
    for customer in destroyed_customers:
        delivery = customer + vrppd.num_customers
        best_insertion = None
        best_total_cost = float('inf')
        for route in best_solution:
            for i in range(0, len(route)):
                for j in range(i, len(route)):
                    # Check if the insertion satisfies the vehicle capacity constraint
                    route_load = sum(vrppd.demands[node - 1] for node in route if node <= vrppd.num_customers)
                    if route_load + vrppd.demands[customer - 1] > vrppd.vehicle_capacity:
                        continue

                    # Create a new solution with the customer and its delivery point inserted
                    new_solution = copy.deepcopy(best_solution)
                    new_route = route[:i] + [customer] + route[i:j] + [delivery] + route[j:]
                    new_solution[best_solution.index(route)] = new_route

                    # Calculate the total cost for the entire solution
                    current_total_cost = vrppd.calculate_total_cost(new_solution)

                    # Check if this is the best insertion so far
                    if current_total_cost < best_total_cost:
                        best_total_cost = current_total_cost
                        best_insertion = (best_solution.index(route), new_route)

        # Apply the best insertion found
        if best_insertion:
            route_index, new_route = best_insertion
            best_solution[route_index] = new_route
            print(new_route)
        else:
            flag = 0

    return best_solution, best_total_cost, flag

def random_repair(solution, destroyed_customers, vrppd):
    new_solution = copy.deepcopy(solution)
    flag = 0
    for customer in destroyed_customers:
        count = 1
        while(count <= 20 and count > 0):
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
            if(count > 20):
                flag = 1

    return new_solution, vrppd.calculate_total_cost(new_solution), flag

def main():
    # 读取数据
    df_test = pd.read_csv('test5.csv')
    demands = df_test['demand'].tolist()
    time_windows = df_test[['start_time', 'end_time']].values.tolist()

    distance_matrix = pd.read_csv('location3.csv', header=None).values

    num_customers = len(demands) // 2  # Adjusted for pickup and delivery
    num_vehicles = int(input("请输入车辆数量: "))
    vehicle_capacity = int(input("请输入每辆车的容量: "))
    penalty_coefficient = 3.1415926

    vrppd = VRPPD(num_customers, num_vehicles, demands, vehicle_capacity, distance_matrix, time_windows, penalty_coefficient)
    print("贪心算法初始解：")
    print_solution(vrppd.routes, vrppd)
    initial_routes = vrppd.routes
    print("初始解总成本：", vrppd.calculate_total_cost(vrppd.routes))

    # 运行ALNS算法
    ALNS(vrppd)
    #print(initial_routes)
    print(vrppd.calculate_total_cost([[0,6,12,13],[0,1,5,7,11,13],[0,2,8,13],[0,3,9,4,10,13],[13]]))
if __name__ == "__main__":
    main()
