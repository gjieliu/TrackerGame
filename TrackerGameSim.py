import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 设置玩家数量和策略数量
num_players = 2
num_strategies = 5

# 设置效用函数参数
E_T_high = np.array([1, 8, 12, 6, 15])
E_T_low = np.array([2, 1, 3, 1, 5])
D_T_high = np.array([8, 10, 6, 12, 4])
D_T_low = np.array([1, 2, 10, 3, 0])

E_A_high = np.array([8, 10, 6, 12, 8])
E_A_low = np.array([1, 3, 1, 5, 2])
D_A_high = np.array([10, 8, 12, 6, 10])
D_A_low = np.array([3, 1, 5, 0, 3])

alpha_T = 0.9
beta_T = 0.1
alpha_A = 0.1
beta_A = 0.9

# 定义效用函数
def utility_function(strategy_profile, player_type):
    if player_type == 'T':
        E_high, E_low = E_T_high, E_T_low
        D_high, D_low = D_T_high, D_T_low
        alpha, beta = alpha_T, beta_T
    else:
        E_high, E_low = E_A_high, E_A_low
        D_high, D_low = D_A_high, D_A_low
        alpha, beta = alpha_A, beta_A
    
    utility = 0
    for i in range(num_strategies):
        if strategy_profile[i] == 1:  # 选择High
            utility += alpha * (E_high[i] * (1 - strategy_profile[num_strategies + i]) + 
                                D_high[i] * strategy_profile[num_strategies + i])
        else:  # 选择Low
            utility += beta * (E_low[i] * (1 - strategy_profile[num_strategies + i]) + 
                               D_low[i] * strategy_profile[num_strategies + i])
    
    return utility

# 定义期望效用函数
def expected_utility(player_strategy, other_player_strategy, player_type):
    expected_utility = 0
    for i in range(num_strategies):
        strategy_profile = np.zeros(2 * num_strategies)
        strategy_profile[i] = player_strategy[i]
        strategy_profile[num_strategies:] = other_player_strategy
        expected_utility += player_strategy[i] * utility_function(strategy_profile, player_type)
    
    return expected_utility

# 定义最优响应函数
def best_response(other_player_strategy, player_type):
    initial_guess = np.random.rand(num_strategies)
    initial_guess /= np.sum(initial_guess)  # 归一化
    bounds = [(0, 1)] * num_strategies
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # 概率之和为1
    
    def objective(player_strategy):
        return -expected_utility(player_strategy, other_player_strategy, player_type)
    
    result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints)
    return result.x

# 计算Nash均衡
def compute_nash_equilibrium():
    theta_T = np.ones(num_strategies) / num_strategies  # 初始化为均匀混合策略
    theta_A = np.ones(num_strategies) / num_strategies
    
    for _ in range(1000):  # 迭代100次寻找均衡
        new_theta_T = best_response(theta_A, 'T')
        new_theta_A = best_response(theta_T, 'A')
        
        if np.allclose(new_theta_T, theta_T) and np.allclose(new_theta_A, theta_A):
            break
        
        theta_T, theta_A = new_theta_T, new_theta_A
    
    return theta_T, theta_A

# 运行仿真并绘制结果
theta_T, theta_A = compute_nash_equilibrium()

print(f"T's Nash Equilibrium Strategy: {theta_T}")
print(f"A's Nash Equilibrium Strategy: {theta_A}")

fig, ax = plt.subplots(figsize=(8, 6))

x = np.arange(num_strategies)  # 策略编号
width = 0.35  # 柱子宽度

ax.bar(x - width/2, theta_T, width, label='Player T', color='blue')
ax.bar(x + width/2, theta_A, width, label='Player A', color='orange')

ax.set_title("Nash Equilibrium Strategies")
ax.set_xlabel("Strategy")
ax.set_ylabel("Probability")
ax.set_xticks(x)
ax.set_xticklabels([f"Strategy {i}" for i in range(num_strategies)])
ax.legend()

plt.tight_layout()
plt.show()
