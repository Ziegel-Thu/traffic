import numpy as np
import pandas as pd
import random
import os
os.chdir('/Users/apple/Desktop/Study/Traffic/Test3')
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

    def initial_solution(self):
        customers = list(range(1, self.num_customers + 1))
        random.shuffle(customers)

        # Insert 0 at the beginning (depot)
        customers = [0] + customers

        routes = [[] for _ in range(self.num_vehicles)]
        vehicle_load = [0] * self.num_vehicles

        current_time = [0] * self.num_vehicles  # Track current time for each vehicle

        for customer in customers:
            best_vehicle = None
            best_cost = float('inf')

            # Find the best vehicle to assign the customer (minimize distance)
            for i in range(self.num_vehicles):
                if vehicle_load[i] + self.demands[customer - 1] <= self.vehicle_capacity:
                    # Calculate cost to pickup
                    pickup_cost = self.distance_matrix[0][customer]
                    if current_time[i] + pickup_cost <= self.time_windows[customer][1]:
                        # Calculate cost to deliver
                        deliver_cost = self.distance_matrix[customer][0]
                        total_cost = pickup_cost + deliver_cost
                        if total_cost < best_cost:
                            best_cost = total_cost
                            best_vehicle = i
            
            # Assign customer to the best vehicle found
            if best_vehicle is not None:
                routes[best_vehicle].append(customer)
                vehicle_load[best_vehicle] += self.demands[customer - 1]
                current_time[best_vehicle] += best_cost

        # Ensure each route ends with depot
        for i in range(self.num_vehicles):
            if routes[i] and routes[i][-1] != 0:
                routes[i].append(0)

        return routes

    def total_distance(self, routes):
        total_distance = 0
        for route in routes:
            if len(route) > 0:
                total_distance += self.distance_matrix[0][route[0]]
                for i in range(1, len(route)):
                    total_distance += self.distance_matrix[route[i - 1]][route[i]]
                total_distance += self.distance_matrix[route[-1]][0]
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
            if customer1 != 0 and customer2 != 0:  # Ensure not swapping depot
                idx1 = new_routes[route1].index(customer1)
                idx2 = new_routes[route2].index(customer2)
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
    # Read data from CSV files
    df_test = pd.read_csv('test.csv')
    demands = df_test['demand'].tolist()
    time_windows = df_test[['start_time', 'end_time']].values.tolist()
    
    distance_matrix = pd.read_csv('location.csv', header=None).values

    num_customers = len(demands)
    
    # Get number of vehicles and vehicle capacity from user input
    num_vehicles = int(input("请输入车辆数量: "))
    vehicle_capacity = int(input("请输入每辆车的容量: "))
    
    penalty_coefficient = 3.14
    
    vrppd = VRPPD(num_customers, num_vehicles, demands, vehicle_capacity, distance_matrix, time_windows, penalty_coefficient)
    initial_temp = 1000
    cooling_rate = 0.995
    stopping_temp = 0.1

    best_solution, best_cost = vrppd.simulated_annealing(initial_temp, cooling_rate, stopping_temp)
    print("最优解：")
    print_solution(best_solution)
    print("最小成本：", best_cost)

if __name__ == "__main__":
    main()
