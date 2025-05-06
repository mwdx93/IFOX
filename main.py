from mealpy.optimizer import Optimizer
import math
import numpy as np

class IFOX(Optimizer):
    def __init__(self, epoch=10000, pop_size=100, mu=3.8, p_levy=0.2, **kwargs):
        super().__init__(**kwargs)
        self.epoch      = self.validator.check_int("epoch",    epoch,    [1, 100000])
        self.pop_size   = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.mu         = mu                  
        self.p_levy     = p_levy              
        self.alpha_min  = 1.0/(0.5*self.epoch)  
        self.half_G     = 0.5 * 9.81         
        self.set_parameters(["epoch","pop_size","mu","p_levy"])

    def levy_flight(self, size, beta=1.5):
        num   = math.gamma(1+beta)*math.sin(math.pi*beta/2)
        den   = beta*math.gamma((1+beta)/2)*2**((beta-1)/2)
        sigma = (num/den)**(1/beta)
        u = self.generator.normal(0, sigma, size)
        v = self.generator.normal(0, 1,     size)
        return u / np.abs(v)**(1/beta)

    def evolve(self, epoch):
        dim = self.g_best.solution.size
        self.alpha = self.alpha_min + (1 - self.alpha_min)* (1 - epoch/self.epoch)
        pop_new = []
        for x in range(len(self.pop)):
            if self.generator.random() < self.alpha:
                beta = self.levy_flight(dim) * self.alpha
            else:
                beta = self.generator.uniform(-self.alpha, self.alpha, size=dim)

            t     = self.generator.random(size=dim)
            jump  = self.half_G * t**2
            dis   = 0.5 * self.g_best.solution

            explore = dis * beta * jump
            exploit = self.g_best.solution + beta * self.alpha

            f1 = self.get_target(explore).fitness
            f2 = self.get_target(exploit).fitness
            if f1 < f2:
                cand, f_cand = explore, f1
            else:
                cand, f_cand = exploit, f2

            if f_cand < self.g_best.target.fitness:
                pos_new = cand
            else:
                pos_new = self.g_best.solution + self.generator.random(size=dim)

            if self.generator.random() < min(beta):
                pos_new = self.problem.lb + self.problem.ub - self.g_best.solution

            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[x] = agent

        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.update_target_for_population(pop_new)


if __name__ == '__main__':
    from opfunu.cec_based.cec2017 import F72017
    from mealpy import FloatVar
    func = F72017()
    problem = {
    "bounds": FloatVar(lb=func.lb, ub=func.ub),
    "obj_func": func.evaluate,
    "minmax": "min",
    "name": "F5",
    "log_to": "console",
    "save_population":True
    }
    optimizer = IFOX(epoch=1000, pop_size=50, name = 'IFOX')
    optimizer.solve(problem)
    
    #Save the results
    optimizer.history.save_diversity_chart()
    optimizer.history.save_runtime_chart()
    optimizer.history.save_trajectory_chart()
    optimizer.history.save_exploration_exploitation_chart()
    optimizer.history.save_global_best_fitness_chart()
    optimizer.history.save_local_best_fitness_chart()
    optimizer.history.save_global_objectives_chart()
    optimizer.history.save_local_objectives_chart()