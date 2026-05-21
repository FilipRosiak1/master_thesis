import numpy as np

from ..base.experiment_islands_model_abc import ExperimentIslands
from ..structures.hall_of_fame import HallOfFame


class ExperimentNumericalIslands(ExperimentIslands):
    def __init__(self, hof_size, popsize, number_of_populations, migration_interval, save_only_best) -> None:
        ExperimentIslands.__init__(self,popsize=popsize,
                                hof_size=hof_size,
                                number_of_populations=number_of_populations,
                                migration_interval=migration_interval,
                                save_only_best=save_only_best)

    def mutate(self, gen):
        return gen + np.random.normal(0, 15, len(gen))

    def cross_over(self, gen1, gen2):
        a = np.random.uniform()
        return a * gen1 + (1.0-a) * gen2

    def evaluate(self, gen):
        return -sum([x*x for x in gen])
