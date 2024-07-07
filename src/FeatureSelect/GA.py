import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, chi2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

num_generations = 30


def GA(X, y, nums, name):
    class GeneticFeatureSelector:
        def __init__(self, n_population=100, n_generations=50, mutation_rate=0.1, crossover_rate=0.8,
                     n_selected_features=200, verbose=False):
            self.n_population = n_population
            self.n_generations = n_generations
            self.mutation_rate = mutation_rate
            self.crossover_rate = crossover_rate
            self.n_selected_features = n_selected_features
            self.verbose = verbose

        def _initialize_population(self, n_features):
            population = np.random.randint(2, size=(self.n_population, n_features))
            return population

        def _calculate_fitness_org(self, X, y, population):
            fitness = []
            for chromosome in population:
                selected_features = [i for i in range(len(chromosome)) if chromosome[i] == 1]
                X_selected = X[:, selected_features]
                selector = SelectKBest(score_func=f_regression, k=min(self.n_selected_features, len(selected_features)))
                selector.fit(X_selected, y)
                fitness.append(-np.mean(selector.scores_))  # Negative mean score as we want to minimize
            return np.array(fitness)

        def _calculate_fitness(self, X, y, population):
            fitness = []
            for chromosome in population:
                features_weight = chromosome
                X_selected = X * features_weight
                selected_features = [i for i in range(len(chromosome)) if chromosome[i] > 0]
                X_selected = X_selected[:, selected_features]
                selector = SelectKBest(score_func=f_regression, k=min(self.n_selected_features, len(selected_features)))
                selector.fit(X_selected, y)
                fitness.append(-np.mean(selector.scores_))  # Negative mean score as we want to minimize
            return np.array(fitness)

        def _selection(self, population, fitness):
            idx = np.argsort(fitness)
            population = population[idx]
            return population[:self.n_population // 2]

        def _crossover(self, population):
            new_population = []
            for _ in range(self.n_population // 2):
                parent1 = population[np.random.choice(range(len(population)))]
                parent2 = population[np.random.choice(range(len(population)))]
                crossover_point = np.random.randint(1, len(parent1) - 1)
                child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
                new_population.extend([child1, child2])
            return np.array(new_population)

        def _mutation(self, population):
            for i in range(len(population)):
                if np.random.rand() < self.mutation_rate:
                    mutation_point = np.random.randint(len(population[i]))
                    population[i, mutation_point] = 1 - population[i, mutation_point]  # Flip the bit
            return population

        def update_feature_weights(self, population, mutation_rate):
            for i in range(len(population)):
                # 对每个个体中的特征权重进行变异操作
                for j in range(len(population[i])):
                    if np.random.rand() < self.mutation_rate:
                        # 根据变异率进行变异操作
                        population[i][j] += np.random.normal(-1, 1)  # 这里使用正态分布进行变异，可以根据实际情况选择其他方式
                        # 限制特征权重的范围
                        population[i][j] = max(0, population[i][j])
            return population

        def select_features(self, X, y):
            n_features = X.shape[1]
            population = self._initialize_population(n_features)
            fit = []
            gene = [x + 1 for x in range(self.n_generations)]
            for generation in range(self.n_generations):
                fitness = self._calculate_fitness(X, y, population)
                population = self._selection(population, fitness)
                population = self._crossover(population)
                population = self._mutation(population)
                if self.verbose:
                    print(f"Generation {generation + 1}: Best Fitness = {np.min(fitness)}")

                fit.append(abs(np.min(fitness)))
                population = self.update_feature_weights(population, self.mutation_rate)

            score = abs(self._calculate_fitness(X, y, population))
            features_weight = np.dot(score.T, population)

            best_fitness_index = np.argmin(self._calculate_fitness(X, y, population))
            best_chromosome = population[best_fitness_index]
            selected_features = np.argsort(-features_weight)[:nums]
            print(name)
            print(np.argsort(-features_weight)[:10] * 0.5 + 400)
            return selected_features, features_weight


    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

    # Initialize and run the genetic feature selector
    selector = GeneticFeatureSelector(n_population=100, n_generations=num_generations, mutation_rate=0.1,
                                      crossover_rate=0.5,
                                      n_selected_features=nums, verbose=True)
    selected_features, features_weight = selector.select_features(X_test, y_test)
    return selected_features


def chromosome_score(chromosome):
    # 将列表中的元素连接成一个字符串
    binary_string = ''.join(str(bit) for bit in chromosome)
    # 将字符串按每7位进行切割
    binary_chunks = [binary_string[i:i + 7] for i in range(0, len(binary_string), 7)]
    # 将每个切割后的字符串转换为二进制数
    decimal_numbers = [int(('0' + chunk), 2) for chunk in binary_chunks]
    return np.array(decimal_numbers)
