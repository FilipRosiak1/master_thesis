from abc import ABC

import numpy as np

from evolalg_steps.base.frams_step import FramsStep
from evolalg_steps.dissimilarity.dissimilarity import Dissimilarity


class FramsDissimilarity(FramsStep):

    def __init__(self, frams_lib, reduction="mean", output_field="dissim",knn=None, *args, **kwargs):
        super(FramsDissimilarity, self).__init__(frams_lib, *args, **kwargs)

        self.output_field = output_field
        self.fn_reduce = Dissimilarity.get_reduction_by_name(reduction)
        self.knn = knn


    def call(self, population):
        super(FramsDissimilarity, self).call(population)
        if len(population) == 0:
            return []
        dissim_matrix = self.frams.dissimilarity([_.genotype for _ in population], 1)
        dissim = Dissimilarity.reduce(dissim_matrix, self.fn_reduce, self.knn)
        for d,ind in zip(dissim, population):
            setattr(ind, self.output_field, d)
        return population
