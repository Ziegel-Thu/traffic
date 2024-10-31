import pandas as pd
import numpy as np

def validate_constraints(z_values, a_values, y_values, Q_values, p_values, u_values, dis, demand, window, N=6, vehicleNum=5):
    tolerance = 1e-4  # 允许的误差范围

    # 验证约束1: 每个任务点必须被访问一次
    for i in range(1, 2 * N + 1):
        expr1 = 0
        for j in range(2 * N + 2):
            for k in range(vehicleNum):
                if i != j:
                    expr1 += z_values.get((i, j, k), 0)
        if not np.isclose(expr1, 1, atol=tolerance):
            print(f"Constraint cons1_{i} violated with value: {expr1}")

    # 验证约束2: 载货点和卸货点的流量平衡
    for i in range(1, N + 1):
        for k in range(vehicleNum):
            expr2 = 0
            for j in range(1, 2 * N + 1):
                if i != j:
                    expr2 += z_values.get((i, j, k), 0)
                    expr2 -= z_values.get((N + i, j, k), 0)
            expr2 += z_values.get((i, 2 * N + 1, k), 0)
            expr2 -= z_values.get((N + i, 2 * N + 1, k), 0)
            if not np.isclose(expr2, 0, atol=tolerance):
                print(f"Constraint cons2_{i}_{k} violated with value: {expr2}")

    # 验证约束3: 每辆车必须从起点出发
    for k in range(vehicleNum):
        expr3 = 0
        for j in range(1, 2 * N + 2):
            expr3 += z_values.get((0, j, k), 0)
        if not np.isclose(expr3, 1, atol=tolerance):
            print(f"Constraint cons3_{k} violated with value: {expr3}")

    # 验证约束4: 对称约束，确保每个任务点进出的平衡
    for i in range(1, 2 * N + 1):
        for k in range(vehicleNum):
            expr4 = 0
            for j in range(2 * N + 2):
                expr4 += z_values.get((i, j, k), 0)
                expr4 -= z_values.get((j, i, k), 0)
            if not np.isclose(expr4, 0, atol=tolerance):
                print(f"Constraint cons4_{i}_{k} violated with value: {expr4}")

    # 验证约束5: 每辆车必须回到终点
    for k in range(vehicleNum):
        expr5 = 0
        for i in range(2 * N + 1):
            expr5 += z_values.get((i, 2 * N + 1, k), 0)
        if not np.isclose(expr5, 1, atol=tolerance):
            print(f"Constraint cons5_{k} violated with value: {expr5}")

    # 验证约束6: 时间约束
    for i in range(2 * N + 2):
        for j in range(2 * N + 2):
            for k in range(vehicleNum):
                if i != j:
                    lhs = a_values.get(j, 0) + (1 - z_values.get((i, j, k), 0)) * 1000000
                    rhs = (a_values.get(i, 0) + dis[i][j]) * z_values.get((i, j, k), 0)
                    if lhs < rhs - tolerance:
                        print(f"Constraint cons6_{i}_{j}_{k} violated with lhs: {lhs}, rhs: {rhs}")

    # 验证约束7: 载货点必须在卸货点之前访问
    for i in range(1, N + 1):
        lhs = a_values.get(N + i, 0)
        rhs = a_values.get(i, 0) + dis[i][N + i]
        if lhs < rhs - tolerance:
            print(f"Constraint cons7_{i} violated with lhs: {lhs}, rhs: {rhs}")

    # 验证约束8: 时间窗口约束
    for j in range(1, 2 * N + 1):
        if y_values.get(j, 0) < -tolerance:
            print(f"Constraint cons8_{j} violated with y: {y_values.get(j, 0)}")

    # 验证约束9: 时间约束的非负性
    for j in range(1, 2 * N + 1):
        if a_values.get(j, 0) < -tolerance:
            print(f"Constraint cons_a_{j} violated with a: {a_values.get(j, 0)}")
    if not np.isclose(a_values.get(0, 0), 0, atol=tolerance):
        print(f"Constraint cons_a_0 violated with a[0]: {a_values.get(0, 0)}")

    # 验证约束10: 超时惩罚约束
    for j in range(N + 1, 2 * N + 1):
        if y_values.get(j, 0) < a_values.get(j, 0) - window[j - (N + 1)] - tolerance:
            print(f"Constraint constr9_{j} violated with y: {y_values.get(j, 0)}, a: {a_values.get(j, 0)}, window: {window[j - (N + 1)]}")
        if p_values.get(j, 0) < a_values.get(j, 0) - window[j - (N + 1)] - 1 - tolerance:
            print(f"Constraint constr10_{j} violated with p: {p_values.get(j, 0)}, a: {a_values.get(j, 0)}, window: {window[j - (N + 1)]}")
        if p_values.get(j, 0) < -tolerance:
            print(f"Constraint constr101_{j} violated with p: {p_values.get(j, 0)}")

    # 验证约束11: 车辆负载约束
    for k in range(vehicleNum):
        for i in range(2 * N + 2):
            for j in range(1, N + 1):
                if i != j and z_values.get((i, j, k), 0) == 1:
                    if not np.isclose(Q_values.get(i, 0) + demand[j - 1], Q_values.get(j, 0), atol=tolerance):
                        print(f"Constraint cons11_{i}_{j}_{k} violated with Q[i]: {Q_values.get(i, 0)}, demand[j-1]: {demand[j - 1]}, Q[j]: {Q_values.get(j, 0)}")

    # 验证约束12: 卸货点的负载平衡
    for k in range(vehicleNum):
        for i in range(2 * N + 2):
            for j in range(N + 1, 2 * N + 1):
                if i != j and z_values.get((i, j, k), 0) == 1:
                    if not np.isclose(Q_values.get(i, 0) - demand[j - N - 1], Q_values.get(j, 0), atol=tolerance):
                        print(f"Constraint cons12_{i}_{j}_{k} violated with Q[i]: {Q_values.get(i, 0)}, demand[j-N-1]: {demand[j - N - 1]}, Q[j]: {Q_values.get(j, 0)}")

    # 验证约束13: 起点的负载约束
    for k in range(vehicleNum):
        for j in range(1, N + 1):
            if z_values.get((0, j, k), 0) == 1:
                if not np.isclose(Q_values.get(j, 0), Q_values.get(0, 0) + demand[j - 1], atol=tolerance):
                    print(f"Constraint cons13_0_{j}_{k} violated with Q[0]: {Q_values.get(0, 0)}, demand[j-1]: {demand[j - 1]}, Q[j]: {Q_values.get(j, 0)}")
        if not np.isclose(Q_values.get(0, 0), 0, atol=tolerance):
            print(f"Constraint cons_q_0_{k} violated with Q[0]: {Q_values.get(0, 0)}")

# 示例调用
# 使用你提供的CSV文件路径读取距离矩阵
dis = pd.read_csv('/Users/apple/Desktop/Study/Traffic/ALNS-additional/location3.csv', header=None).values

# 确保 z, y, p 没有数据时初始化为 0
z_values = {(i, j, k): 0 for i in range(14) for j in range(14) for k in range(5)}
y_values = {i: 0 for i in range(14)}
p_values = {i: 0 for i in range(14)}

# 根据实际路径填充值
z_values.update({
    (0, 1, 0): 1, (1, 5, 0): 1, (5, 7, 0): 1, (7, 11, 0): 1, (11, 2, 0): 1, (2, 8, 0): 1, (8, 13, 0): 1,
    (0, 3, 1): 1, (3, 9, 1): 1, (9, 6, 1): 1, (6, 4, 1): 1, (4, 10, 1): 1, (10, 12, 1): 1, (12, 13, 1): 1,
    (0, 13, 2): 1, (0, 13, 3): 1, (0, 13, 4): 1
})

a_values = {
    1: 1.15327162, 5: 1.78000511, 7: 3.75411415, 11: 4.61157377, 2: 5.32472996, 8: 7.17688735, 13: 8.92742291,3: 1.64039749, 9: 2.48276624, 6: 2.85147667, 4: 3.23989497, 10: 4.88777589, 12: 6.52708776
}

y_values.update({
    8: 0.69916998,
})

Q_values = {
    1: 10, 5: 40, 7: 30, 11: 0, 2: 20, 8: 0, 13: 0,3: 30, 9: 0, 6: 30, 4: 50, 10: 30, 12: 0,
}

p_values.update({
    8: 2.19650722,
})

u_values = {
    0: 1, 1: 1, 2: 0, 3: 0, 4: 0,
}

demand = [10, 20, 30, 20, 30, 30, 10, 20, 30, 20, 30, 30]
window = [1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 4.836524177878462, 6.477717374902587, 4.346499155632049, 6.030109860947896, 5.314449151747034, 6.894066036835748, 1000000]

validate_constraints(z_values, a_values, y_values, Q_values, p_values, u_values, dis, demand, window, N=6, vehicleNum=5)
