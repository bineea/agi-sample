import math
import random
import time
from decimal import Decimal


def find_combinations(nums, target):
    nums.sort()
    dp = {0: [[]]}  # 初始化dp字典，键为和，值为组合列表

    for num in nums:
        new_dp = dp.copy()
        for t in dp:
            if t + num <= target:
                if t + num not in new_dp:
                    new_dp[t + num] = []
                for comb in dp[t]:
                    new_dp[t + num].append(comb + [num])
        dp = new_dp

    return dp.get(target, [])


# 最快的方法
def find_combinations_backtrack(nums, target):
    result = []
    nums.sort()

    def backtrack(start, path, target):
        if target == 0:
            result.append(path)
            return
        for i in range(start, len(nums)):
            if nums[i] > target:
                break
            backtrack(i + 1, path + [nums[i]], target - nums[i])

    backtrack(0, [], target)
    return result


def find_combinations_dp(nums, target):
    n = len(nums)
    rounded_nums = [round(Decimal(num), 2) for num in nums]
    rounded_target = round(Decimal(target), 2)

    dp = [[False] * (int(rounded_target) + 1) for _ in range(n + 1)]
    dp[0][0] = True

    for i in range(1, n + 1):
        num = rounded_nums[i - 1]
        for j in range(int(rounded_target), int(num) - 1, -1):
            dp[i][j] = dp[i - 1][j] or (j >= int(num) and dp[i - 1][j - int(num)])

    if not dp[n][int(rounded_target)]:
        return []

    result = []
    i, j = n, int(rounded_target)
    while i > 0 and j > 0:
        if dp[i - 1][j]:
            i -= 1
        else:
            result.append(rounded_nums[i - 1])
            j -= int(rounded_nums[i - 1])
            i -= 1

    return result


def simulated_annealing(nums, target, max_iter=10000, initial_temp=1000, cooling_rate=0.99):
    def get_sum(combination):
        return sum(combination)

    def get_random_combination(nums):
        return [num for num in nums if random.choice([True, False])]

    current_combination = get_random_combination(nums)
    current_sum = get_sum(current_combination)
    best_combination = current_combination
    best_sum = current_sum

    temperature = Decimal(initial_temp)

    for _ in range(max_iter):
        if temperature <= 0:
            break

        new_combination = get_random_combination(nums)
        new_sum = get_sum(new_combination)

        if abs(new_sum - target) < abs(current_sum - target):
            current_combination = new_combination
            current_sum = new_sum
            if abs(current_sum - target) < abs(best_sum - target):
                best_combination = current_combination
                best_sum = current_sum
        else:
            acceptance_prob = math.exp(-(abs(new_sum - target) - abs(current_sum - target)) / Decimal(temperature))
            if random.random() < acceptance_prob:
                current_combination = new_combination
                current_sum = new_sum

        temperature *= Decimal(cooling_rate)

    return best_combination if abs(best_sum - target) < Decimal('500') else []


def simulated_annealing_test(nums, target, initial_temperature=1000, cooling_rate=0.99, max_iterations=1000):
    # 将数字四舍五入到小数点后两位
    rounded_nums = [round(Decimal(num), 2) for num in nums]
    rounded_target = round(Decimal(target), 2)

    # 初始化解
    current_solution = [random.choice(rounded_nums) for _ in range(len(rounded_nums))]
    current_fitness = abs(sum(current_solution) - rounded_target)

    temperature = initial_temperature
    for _ in range(max_iterations):
        # 生成新的解
        new_solution = current_solution.copy()
        new_solution[random.randint(0, len(new_solution) - 1)] = random.choice(rounded_nums)

        # 计算新解的适应度
        new_fitness = abs(sum(new_solution) - rounded_target)

        # 决定是否接受新解
        if new_fitness < current_fitness:
            current_solution = new_solution
            current_fitness = new_fitness
        elif random.random() < math.exp(-(new_fitness - current_fitness) / Decimal(temperature)):
            current_solution = new_solution
            current_fitness = new_fitness

        # 降温
        temperature *= cooling_rate

        # 检查是否找到满足条件的解
        if round(sum(current_solution), 2) == rounded_target:
            return current_solution

    # 如果找不到满足条件的解,返回空列表
    return []


def fitness_function(nums, target, solution):
    return abs(sum(solution) - target)

def crossover(parent1, parent2):
    child = parent1.copy()
    for i in range(len(parent1)):
        if random.random() < 0.5:
            child[i] = parent2[i]
    return child

def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.choice(nums)
    return individual

def genetic_algorithm(nums, target, population_size=100, num_generations=100, mutation_rate=0.1):
    # 将数字四舍五入到小数点后两位
    rounded_nums = [round(Decimal(num), 2) for num in nums]
    rounded_target = round(Decimal(target), 2)

    # 初始化种群
    population = [[random.choice(rounded_nums) for _ in range(len(rounded_nums))] for _ in range(population_size)]

    for generation in range(num_generations):
        # 计算适应度
        fitness_scores = [fitness_function(rounded_nums, rounded_target, individual) for individual in population]

        # 选择父代
        parents = [population[i] for i in sorted(range(len(fitness_scores)), key=lambda x: fitness_scores[x])[:2]]

        # 交叉和变异
        children = [crossover(parents[0], parents[1]) for _ in range(population_size - 2)]
        children.append(parents[0])
        children.append(parents[1])
        for i in range(population_size):
            population[i] = mutate(children[i], mutation_rate)

        # 检查是否找到满足条件的解
        for individual in population:
            if round(sum(individual), 2) == rounded_target:
                return individual

    # 如果找不到满足条件的解,返回空列表
    return []


if __name__ == "__main__":
    nums = [Decimal('69630.00'), Decimal('795544.20'), Decimal('2750.00'), Decimal('8250.00'), Decimal('79386.56'),
            Decimal('77699.60'), Decimal('26180.00'), Decimal('279400.00'), Decimal('67.21'), Decimal('43.78'),
            Decimal('269.72'), Decimal('253704.00'), Decimal('19096.00'), Decimal('223.30'), Decimal('31.90'),
            Decimal('21.89'), Decimal('545.60'), Decimal('134.86'), Decimal('1028.50'), Decimal('5428.50'),
            Decimal('100000.1'), Decimal('100000.2'), Decimal('100000.3'), Decimal('100000.4'), Decimal('100000.5'),
            Decimal('100000.6'), Decimal('100000.7'), Decimal('100000.8'), Decimal('100000.9'), Decimal('100001.1'),
            Decimal('100000.6'), Decimal('100000.7'), Decimal('100000.8'), Decimal('100000.9'), Decimal('100001.1'),
            Decimal('100000.6'), Decimal('100000.7'), Decimal('100000.8'), Decimal('100000.9'), Decimal('100001.1'),
            Decimal('100000.6'), Decimal('100000.7'), Decimal('100000.8'), Decimal('100000.9'), Decimal('100001.1'),
            Decimal('100000.6'), Decimal('100000.7'), Decimal('100000.8'), Decimal('100000.9'), Decimal('100001.1'),
            Decimal('100000.6'), Decimal('100000.7'), Decimal('100000.8'), Decimal('100000.9'), Decimal('100001.1'),
            Decimal('1000.01'), Decimal('2001.21'), Decimal('33.56'), Decimal('27.32'), Decimal('998'),
            Decimal('62469.76')
            ]
    target = Decimal('1619435.62')

    # nums = [Decimal('31.9'), Decimal('21.89'), Decimal('545.6'), Decimal('134.86'), Decimal('1028.50'),
    #         Decimal('5428.5'), Decimal('62469.76')]
    # target = Decimal('69661.01')

    print(sum(nums))

    start_time = time.time()
    print(f"开始时间: {start_time}")
    # combinations = find_combinations_backtrack(nums, target)
    # combinations = find_combinations(nums, target)
    combinations = simulated_annealing(nums, target)
    # combinations = genetic_algorithm(nums, target)
    print(combinations)
    print(f"执行时间: {time.time() - start_time}")

