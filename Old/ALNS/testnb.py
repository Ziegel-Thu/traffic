import numpy as np
import pandas as pd
import random
import math
import os
import copy

os.chdir('/Users/apple/Desktop/Study/Traffic/Test5')
Beta=[]
class VRPPD:
    def __init__(self, num_customers, num_vehicles, demands, vehicle_capacity, distance_matrix, time_windows, penalty_coefficient):
        self.num_customers = num_customers
        self.num_vehicles = num_vehicles
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        self.distance_matrix = distance_matrix
        self.time_windows = time_windows
        self.penalty_coefficient = penalty_coefficient
        '''self.routes=[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],[21],[21]]'''
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

def ALNS(vrppd, max_iterations=100, destruction_level=0.3):
    current_solution = vrppd.routes
    best_solution = current_solution
    best_cost = vrppd.total_distance(current_solution) + vrppd.total_penalty(current_solution)
    temperature = 1000.0
    cooling_rate = 0.999
    
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        print(f"Iteration {iteration}")
        
        # Perform worst destroy
        destroyed_customers, new_solution = worst_destroy(current_solution, vrppd, destruction_level)
        print(f"第 {iteration} 次迭代后的新解（破坏后）：{new_solution}")
        
        # Perform repair
        repaired_solution, new_cost = repair(new_solution, destroyed_customers, vrppd)
        print(f"第 {iteration} 次迭代后的当前解（修复后）：{repaired_solution}")
        print(f"Current Solution Cost: {new_cost}")
        # Acceptance criterion
        if new_cost < best_cost :
            current_solution = repaired_solution
            best_cost = new_cost
            best_solution=current_solution
        
        temperature *= cooling_rate

        print(f"Best Solution Cost: {best_cost}")
        if new_cost<14:
            Beta.append(iteration)
    
    print("\nFinal Solution:")
    print_solution(best_solution, vrppd)
    print("Final Solution Cost:", best_cost)

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
        for i in range(1, len(route) - 1):
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
                        '''if delivery_index>5:
                            print(delivery_index)
                            print(route)
                            print(delivery)'''
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
            if len(best_route) > 2 and best_route[-1] !=2 * vrppd.num_customers + 1:  # Ensure the route ends at the depot
                best_route.append(2 * vrppd.num_customers + 1)
                '''print('Beta!')
                print(best_route)'''
                # Return to depot

    print("破坏后移除的客户：", remove_customers)
    print("破坏后的新解：", new_solution)
    return remove_customers, new_solution

def repair(solution, destroyed_customers, vrppd):
    best_solution=vrppd.routes
    new_solution = copy.deepcopy(solution)
    print("修复前的新解：", new_solution)
    best_cost = float('inf')
    best_total=float('inf')
    k=0
    for customer in destroyed_customers:
        delivery = customer + vrppd.num_customers


        for route in new_solution:
            for i in range(len(route)):
                for j in range(i, len(route) ):
                    new_route = route[:i] + [customer] + route[i:j] + [delivery] + route[j:]
                    current_cost = vrppd.total_distance([new_route]) + vrppd.total_penalty([new_route])
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_insertion = (route, i, j)

                    elif random.random() < math.exp((best_cost - current_cost) / 1000.0):
                        best_cost = current_cost
                        best_insertion = (route, i, j)

        if best_insertion:
            route, i, j = best_insertion
            route[i:i] = [customer]
            route[j+1:j+1] = [delivery]
        print("修复中的新解：", new_solution)
        k+=1
        print(k)
        if k==len(destroyed_customers):
            print(1111)
            new_total=vrppd.total_distance(new_solution)+vrppd.total_penalty(new_solution)
            print(best_cost)
            if best_total>new_total:
                print(2222)
                best_total=new_total
                best_solution=new_solution

                
    print("修复后的新解：", best_solution)
    return best_solution, best_total

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
    print(Beta)
if __name__ == "__main__":
    main()