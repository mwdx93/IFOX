from mealpy.optimizer import Optimizer


class IFOX(Optimizer):
    def __init__(self, epoch=10000, pop_size=100, mu=3.8, p_levy=0.2, **kwargs):
        super().__init__(**kwargs)
        self.epoch      = self.validator.check_int("epoch",    epoch,    [1, 100000])
        self.pop_size   = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.mu         = mu                  # chaotic map parameter
        self.p_levy     = p_levy              # prob of injecting Lévy-flight
        self.alpha_min  = 1.0/(0.5*self.epoch)  # floor for alpha
        self.half_G     = 0.5 * 9.81          # gravity constant
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

        # 1) update α with a logistic (chaotic) map
        self.alpha = self.alpha_min + (1 - self.alpha_min)* (1 - epoch/self.epoch)
  

        pop_new = []
        for agent in self.pop:
            x = agent.solution

            # 2) draw β either by Alpha or Lévy
            if self.generator.random() < self.alpha:
                beta = self.levy_flight(dim) * self.alpha
            else:
                beta = self.generator.uniform(-self.alpha, self.alpha, size=dim)

            # 3) compute original IFOX jump and dis
            t     = self.generator.random(size=dim)
            jump  = self.half_G * t**2
            dis   = 0.5 * self.g_best.solution

            # 4) original IFOX formulas
            explore = dis * beta * jump
            exploit = self.g_best.solution + beta * self.alpha

            # 5) pick the better of the two
            f1 = self.get_target(explore).fitness
            f2 = self.get_target(exploit).fitness
            if f1 < f2:
                cand, f_cand = explore, f1
            else:
                cand, f_cand = exploit, f2

            # 6) elitism vs g_best
            if f_cand < self.g_best.target.fitness:
                pos_new = cand
            else:
                # small random move around g_best to escape stagnation
                pos_new = self.g_best.solution + self.generator.random(size=dim)

            # 7) occasionally try an opposition move on g_best
            if self.generator.random() < min(beta):
                pos_new = self.problem.lb + self.problem.ub - self.g_best.solution


                    # Compare with global best (elitism)
            pos_new = self.correct_solution(pos_new)
            agent = self.generate_empty_agent(pos_new)
            pop_new.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                agent.target = self.get_target(pos_new)
                self.pop[x] = agent

        if self.mode in self.AVAILABLE_MODES:
            self.pop = self.update_target_for_population(pop_new)
