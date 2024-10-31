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
        self.routes=[[8, 9, 4, 19, 2, 18, 14, 12, 21],[5, 10, 7, 17, 15, 20, 21],[3, 1, 13, 11, 6, 16, 21]]
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

def ALNS(vrppd, max_iterations=10000, destruction_level=0.3):
    current_solution = vrppd.routes
    best_solution = current_solution
    best_cost = vrppd.total_distance(current_solution) + vrppd.total_penalty(current_solution)
    temperature = 10000.0
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
        if new_cost<19:
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
                        if pickup_index==0:
                            node0=0
                            node1=route[delivery_index + 1]
                            fitness = distance_matrix[node0][node1] - distance_matrix[node0][pickup] - distance_matrix[pickup][delivery] - distance_matrix[delivery][node1]
                            
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
                # Return to depot

    print("破坏后移除的客户：", remove_customers)
    print("破坏后的新解：", new_solution)
    return remove_customers, new_solution

def repair(solution, destroyed_customers, vrppd, temperature=1.0,cooling_rate=0.3):
    best_solution = copy.deepcopy(solution)
    best_total_cost = float('inf')

    for customer in destroyed_customers:
        delivery = customer + vrppd.num_customers
        best_insertion = None
        best_total_cost=float('inf')
        for route in best_solution:
            for i in range(1, len(route)):
                for j in range(i, len(route)):
                    # Check if the insertion satisfies the vehicle capacity constraint
                    route_load = sum(vrppd.demands[node - 1] for node in route if node <= vrppd.num_customers)
                    if route_load + vrppd.demands[customer - 1] > vrppd.vehicle_capacity:
                        continue

                    # Create a new solution with the customer and its delivery point inserted
                    new_solution = copy.deepcopy(best_solution)
                    new_route = route[:i] + [customer] + route[i:j] + [delivery] + route[j:]
                    new_solution[best_solution.index(route)] = new_route

                    # Calculate the total distance and penalty for the entire solution
                    current_total_distance = vrppd.total_distance(new_solution)
                    current_total_penalty = vrppd.total_penalty(new_solution)
                    current_total_cost = current_total_distance + current_total_penalty

                    # Simulated annealing acceptance criterion
                    cost_diff = current_total_cost - best_total_cost
                    if cost_diff < 0 or random.random() < math.exp(-cost_diff / temperature):
                        '''if cost_diff > 0:
                            print('yes')
                            print(cost_diff)
                            print(temperature)
                            print(math.exp(-cost_diff / temperature))'''
                        best_total_cost = current_total_cost
                        best_insertion = (best_solution.index(route), new_route)
            '''print(best_insertion)
            print(route)
            print(best_total_cost)
            print(111)'''
        # Apply the best insertion found
        #print(234)
        if best_insertion:
            route_index, new_route = best_insertion
            best_solution[route_index] = new_route

        # Reduce the temperature
        temperature *= cooling_rate

        #print(best_solution)

    print("修复后的新解：", best_solution)
    return best_solution, best_total_cost



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
    #ALNS(vrppd)
    print(Beta)
    print(vrppd.total_distance([[21],[ 8, 9, 4, 6, 2, 18, 14, 16, 19, 12, 21],[1, 5, 10, 7, 11, 17, 20, 15, 3, 13, 21]])+vrppd.total_penalty([[21],[ 8, 9, 4, 6, 2, 18, 14, 16, 19, 12, 21],[1, 5, 10, 7, 11, 17, 20, 15, 3, 13, 21]]))
if __name__ == "__main__":
    main()
