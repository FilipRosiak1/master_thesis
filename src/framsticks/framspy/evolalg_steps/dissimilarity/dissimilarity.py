from abc import ABC

from evolalg_steps.base.step import Step
import numpy as np


class Dissimilarity(Step, ABC):

    def __init__(self, reduction="mean", output_field="dissim", knn=None, *args, **kwargs):
        super(Dissimilarity, self).__init__(*args, **kwargs)

        self.output_field = output_field
        self.fn_reduce = Dissimilarity.get_reduction_by_name(reduction)
        self.knn = knn


    @staticmethod
    def reduce(dissim_matrix, fn_reduce, knn):
        if fn_reduce is None:
            return dissim_matrix
        elif fn_reduce is Dissimilarity.knn_mean:
            return fn_reduce(dissim_matrix, 1, knn)
        else:
            return fn_reduce(dissim_matrix, axis=1)


    @staticmethod
    def knn_mean(dissim_matrix, axis, knn):
        return np.mean(np.partition(dissim_matrix, knn)[:, :knn], axis=axis)


    @staticmethod
    def get_reduction_by_name(reduction: str):

        if reduction not in REDUCTION_FUNCTION:
            raise ValueError(f"Unknown reduction type '{reduction}'. Supported: {','.join(REDUCTION_FUNCTION.keys())}")

        return REDUCTION_FUNCTION[reduction]



REDUCTION_FUNCTION = {
            "mean": np.mean,
            "max": np.max,
            "min": np.min,
            "sum": np.sum,
            "knn_mean": Dissimilarity.knn_mean,
            "none": None
        }
