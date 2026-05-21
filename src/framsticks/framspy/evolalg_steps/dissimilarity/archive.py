import copy
from typing import Union

from evolalg_steps.base.step import Step
from evolalg_steps.dissimilarity.dissimilarity import Dissimilarity
from evolalg_steps.dissimilarity.frams_dissimilarity import FramsDissimilarity
from evolalg_steps.statistics.halloffame_custom import HallOfFameCustom

# TODO not fully tested. Verify if works OK, in particular test adding new individuals.
class ArchiveDissimilarity(Step):

    def __init__(self, archive_size, dissim: Union[Dissimilarity, FramsDissimilarity], order="max", field="dissim"):
        self.name= "archive"
        self.archive_size = archive_size
        self.archive = []
        self.dissim = dissim
        self.order = order
        self.field = field
        if self.order not in ["min", "max"]:
            raise ValueError("Order must be min or max")

        if self.archive_size < 0:
            raise ValueError(f"Archive size must be integer greater than or equal to 0. Got {self.archive_size}")

    def call(self, population):
        super(ArchiveDissimilarity, self).call(population)
        population_archive = population + self.archive
        population_archive = self.dissim(population_archive)

        population = population_archive[:len(population)]
        order = 1
        if self.order == "max":
            order *= -1
        sorted_archive = sorted(population_archive, key=lambda x: getattr(x, self.field) * order)
        self.archive = copy.deepcopy(sorted_archive[:self.archive_size])
        return population
