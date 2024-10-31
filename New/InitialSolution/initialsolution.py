import pandas as pd

class VRPPD:
    def __init__(self, num_customers, num_vehicles, demands, vehicle_capacity, distance_matrix, time_windows, penalty_coefficient):
        self.num_customers = num_customers
        self.num_vehicles = num_vehicles
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        self.distance_matrix = distance_matrix
        self.time_windows = time_windows
        self.penalty_coefficient = penalty_coefficient
        self.routes = self.greedy_initial_solution()
        print("初始解生成完成")
        print("初始解：", self.routes)

    def greedy_initial_solution(self):
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

# 这里可以添加其他辅助函数或类
