import numpy as np
import pandas as pd
import random
import math
import os
import copy

os.chdir('/Users/apple/Desktop/Study/Traffic/ALNS')

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
            if routes[i] and routes[i][-1] != final_depot:
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

def ALNS(vrppd, max_iterations=1000, destruction_level=0.2):
    current_solution = vrppd.routes
    best_solution = current_solution
    best_cost = vrppd.total_distance(current_solution) + vrppd.total_penalty(current_solution)
    
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        print(f"Iteration {iteration}")
        
        # Perform worst destroy
        destroyed_customers, new_solution = worst_destroy(current_solution, vrppd, destruction_level)
        print(f"第 {iteration} 次迭代后的新解（破坏后）：{new_solution}")
        
        # Perform repair
        repaired_solution = repair(new_solution, destroyed_customers, vrppd)
        print(f"第 {iteration} 次迭代后的当前解（修复后）：{repaired_solution}")
        
        # Evaluate new solution
        current_cost = vrppd.total_distance(repaired_solution) + vrppd.total_penalty(repaired_solution)
        
        # Acceptance criterion
        if current_cost < best_cost:
            best_solution = repaired_solution
            best_cost = current_cost
        
        print(f"Current Solution Cost: {current_cost}")
        print(f"Best Solution Cost: {best_cost}")
    
    print("\nFinal Solution:")
    print_solution(best_solution, vrppd)
    print("Final Solution Cost:", best_cost)

def worst_destroy(solution, vrppd, destruction_level):
    remove_customers = []
    new_solution = copy.deepcopy(solution)
    distance_matrix = vrppd.distance_matrix
    destroy_num = math.ceil(destruction_level * len([customer for route in new_solution for customer in route]))

    while len(remove_customers) < destroy_num:
        best_fitness = float("inf")
        best_customer_pair = None
        best_route = None

        for route in new_solution:
            for i in range(1, len(route) - 1):
                pickup = route[i]
                if isinstance(pickup, int) and pickup <= vrppd.num_customers:  # Ensure it's a pickup node
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
                            best_customer_pair = (pickup, delivery)
                            best_route = route

        if best_customer_pair:
            pickup, delivery = best_customer_pair
            best_route.remove(pickup)
            best_route.remove(delivery)
            remove_customers.append(pickup)
            remove_customers.append(delivery)
            if len(best_route) > 2 and best_route[-1] != best_route[-2]:  # Ensure the route ends at the depot
                best_route.append(2 * vrppd.num_customers + 1)  # Return to depot

    return remove_customers, new_solution

def repair(solution, destroyed_customers, vrppd):
    new_solution = copy.deepcopy(solution)
    for customer in destroyed_customers:
        best_cost = float('inf')
        best_insertion = None
        for route in new_solution:
            for i in range(1, len(route)):
                for j in range(i, len(route) + 1):
                    new_route = route[:i] + [customer] + route[i:j] + [customer + vrppd.num_customers] + route[j:]
                    if vrppd.total_distance([new_route]) < best_cost:
                        best_cost = vrppd.total_distance([new_route])
                        best_insertion = (route, i, j)
        if best_insertion:
            route, i, j = best_insertion
            route[i:i] = [customer]
            route[j+1:j+1] = [customer + vrppd.num_customers]
    return new_solution

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
    print("初始解总成本：", vrppd.total_distance(vrppd.routes) + vrppd.total_penalty(vrppd.routes))
    # 运行ALNS算法
    ALNS(vrppd)

if __name__ == "__main__":
    main()
