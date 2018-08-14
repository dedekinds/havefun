import numpy
import matplotlib.pyplot as plt


data = numpy.array([[77, 92],
                    [22, 22],
                    [29, 87],
                    [50, 46],
                    [99, 90]])


class GA(object):
    """
    遗传算法解决0-1背包问题
    """

    def __init__(self, length, number, iter_number):
        """
        参数初始化
        :param length: 5
        :param number: 300
        :param iter_number: 300
        """
        self.length = length  # 确定染色体编码长度
        self.number = number  # 确定初始化种群数量
        self.iteration = iter_number  # 设置迭代次数
        self.bag_capacity = 100  # 背包容量

        self.retain_rate = 0.2  # 每一代精英选择出前20%
        self.random_selection_rate = 0.5  # 对于不是前20%的，有0.5的概率可以进行繁殖
        self.mutation_rate = 0.01  # 变异概率0.01

    def initial_population(self):
        """
        种群初始化，

        :return: 返回种群集合
        """
        init_population = numpy.random.randint(low=0, high=2, size=[self.length, self.number], dtype=numpy.int16)
        return init_population

    def weight_price(self, chromosome):
        """
        计算累计重量和累计价格
        :param chromosome:
        :return:返回每一个个体的累计重量和价格
        """
        w_accumulation = 0
        p_accumulation = 0
        for i in range(len(chromosome)):

            w = chromosome[i]*data[i][0]
            p = chromosome[i]*data[i][1]
            w_accumulation = w + w_accumulation
            p_accumulation = p + p_accumulation

        return w_accumulation, p_accumulation

    def fitness_function(self, chromosome):
        """
        计算适应度函数，一般来说，背包的价值越高越好，但是
        当重量超过100时，适应度函数=0
        :param chromosome:
        :return:
        """

        weight, price = self.weight_price(chromosome)
        if weight > self.bag_capacity:
            fitness = 0
        else:
            fitness = price

        return fitness

    def fitness_average(self, init_population):
        """
        求出这个种群的平均适应度，才能知道种群已经进化好了
        :return:返回的是一个种群的平均适应度
        """
        f_accumulation = 0
        for z in range(init_population.shape[1]):
            f_tem = self.fitness_function(init_population[:, z])
            f_accumulation = f_accumulation + f_tem
        f_accumulation = f_accumulation/init_population.shape[1]
        return f_accumulation

    def selection(self, init_population):
        """
        选择
        :param init_population:
        :return: 返回选择后的父代，数量是不定的
        """
        sort_population = numpy.array([[], [], [], [], [], []])  # 生成一个排序后的种群列表，暂时为空
        for i in range(init_population.shape[1]):

            x1 = init_population[:, i]
            # print('打印x1', x1)
            x2 = self.fitness_function(x1)
            x = numpy.r_[x1, x2]
            # print('打印x', x)
            sort_population = numpy.c_[sort_population, x]

        sort_population = sort_population.T[numpy.lexsort(sort_population)].T  # 联合排序，从小到大排列

        # print('排序后长度', sort_population.shape[1])
        print(sort_population)

        # 选出适应性强的个体，精英选择
        retain_length = sort_population.shape[1]*self.retain_rate

        parents = numpy.array([[], [], [], [], [], []])  # 生成一个父代列表，暂时为空
        for j in range(int(retain_length)):
            y1 = sort_population[:, -(j+1)]
            parents = numpy.c_[parents, y1]

        # print(parents.shape[1])

        rest = sort_population.shape[1] - retain_length  # 精英选择后剩下的个体数
        for q in range(int(rest)):

            if numpy.random.random() < self.random_selection_rate:
                y2 = sort_population[:, q]
                parents = numpy.c_[parents, y2]

        parents = numpy.delete(parents, -1, axis=0)  # 删除最后一行，删除了f值
        # print('打印选择后的个体数')
        # print(parents.shape[0])

        parents = numpy.array(parents, dtype=numpy.int16)

        return parents

    def crossover(self, parents):
        """
        交叉生成子代，和初始化的种群数量一致
        :param parents:
        :return:返回子代
        """
        children = numpy.array([[], [], [], [], []])  # 子列表初始化

        while children.shape[1] < self.number:
            father = numpy.random.randint(0, parents.shape[1] - 1)
            mother = numpy.random.randint(0, parents.shape[1] - 1)
            if father != mother:
                # 随机选取交叉点
                cross_point = numpy.random.randint(0, self.length)
                # 生成掩码，方便位操作
                mark = 0
                for i in range(cross_point):
                    mark |= (1 << i)

                father = parents[:, father]
                # print(father)
                mother = parents[:, mother]

                # 子代将获得父亲在交叉点前的基因和母亲在交叉点后（包括交叉点）的基因
                child = ((father & mark) | (mother & ~mark)) & ((1 << self.length) - 1)

                children = numpy.c_[children, child]

                # 经过繁殖后，子代的数量与原始种群数量相等，在这里可以更新种群。
                # print('子代数量', children.shape[1])
        # print(children.dtype)
        children = numpy.array(children, dtype=numpy.int16)
        return children

    def mutation(self, children):
        """
        变异

        :return:
        """
        for i in range(children.shape[1]):

            if numpy.random.random() < self.mutation_rate:
                j = numpy.random.randint(0, self.length - 1)  # s随机产生变异位置
                children[:, i] ^= 1 << j  # 产生变异
        children = numpy.array(children, dtype=numpy.int16)
        return children

    def plot_figure(self, iter_plot, f_plot, f_set_plot):
        """
        画出迭代次数和平均适应度曲线图
        画出迭代次数和每一步迭代最大值图
        :return:
        """
        plt.figure()

        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)

        plt.sca(ax1)
        plt.plot(iter_plot, f_plot)
        plt.ylim(0, 140)  # 设置y轴范围

        plt.sca(ax2)
        plt.plot(iter_plot, f_set_plot)
        plt.ylim(0, 140)  # 设置y轴范围
        plt.show()

    def main(self):
        """
        main函数,用来进化
        对当前种群依次进行选择、交叉并生成新一代种群，然后对新一代种群进行变异
        :return:
        """
        init_population = self.initial_population()
        # print(init_population)

        iter_plot = []
        f_plot = []
        iteration = 0

        f_set_plot = []

        while iteration < self.iteration:  # 设置迭代次数300

            parents = self.selection(init_population)  # 选择后的父代
            children = self.crossover(parents)
            mutation_children = self.mutation(children)

            init_population = mutation_children

            f_set = []  # 求出每一步迭代的最大值
            for init in range(init_population.shape[1]):
                f_set_tem = self.fitness_function(init_population[:, init])
                f_set.append(f_set_tem)

            f_set = max(f_set)

            f_set_plot.append(f_set)

            iter_plot.append(iteration)
            iteration = iteration+1
            print("第%s进化得如何******************************************" % iteration)
            f_average = self.fitness_average(init_population)
            f_plot.append(f_average)
            print(f_set)
            # f_accumulation = f_accumulation + f
            # f_print = f_accumulation/(iteration + 1)
            # print(f_print)
        self.plot_figure(iter_plot, f_plot, f_set_plot)


if __name__ == '__main__':
    g1 = GA(5, 300, 100)
    g1.main()