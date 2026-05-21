import argparse
import sys
import numpy as np

from FramsticksLib import FramsticksLib, DissimMethod

from ..frams_base.experiment_frams_niching import ExperimentFramsNiching
from ..frams_base.experiment_frams_islands import ExperimentFramsIslands
from ..numerical_example.numerical_example import ExperimentNumerical
from ..numerical_example.numerical_islands_example import ExperimentNumericalIslands
from ..structures.individual import Individual

from ..utils import ensureDir


GENERATIONS = 10

SETTINGS_TO_TEST_NUMERIC = {
    'hof_size': [0, 10],
    'popsize': [8],
    'archive': [8],
    'pmut': [0.7],
    'pxov': [0.2],
    'tournament': [5],
    'initialgenotype':[np.array([100, 100, 100, 100]), np.array([-100,-100])]
}

SETTINGS_TO_TEST_NUMERIC_ISLAND = {
    'hof_size': [0, 10],
    'popsize': [8],
    'archive': [8],
    'pmut': [0.7],
    'pxov': [0.2],
    'tournament': [5],
    'migration_interval': [1,5],
    'number_of_populations':[1,5],
    'initialgenotype':[np.array([100, 100, 100, 100]), np.array([-100,-100])]
}

SETTINGS_TO_TEST_FRAMS_NICHING = {
    'opt': ['velocity', 'vertpos'],
    'max_numparts': [None],
    'max_numjoints': [20],
    'max_numneurons': [20],
    'max_numconnections': [None],
    'max_numgenochars': [20],
    'hof_size': [0, 10],
    'normalize': ['none', 'max', 'sum'],
    'dissim': [DissimMethod.GENE_LEVENSHTEIN],# DissimMethod.PHENE_STRUCT_OPTIM, DissimMethod.PHENE_DESCRIPTORS, DissimMethod.PHENE_DENSITY_COUNT, DissimMethod.PHENE_DENSITY_FREQ],
    'crowding_dissim': [DissimMethod.FITNESS, DissimMethod.GENE_LEVENSHTEIN, DissimMethod.PHENE_STRUCT_OPTIM],#, DissimMethod.PHENE_STRUCT_OPTIM, DissimMethod.PHENE_DESCRIPTORS, DissimMethod.PHENE_DENSITY_COUNT, DissimMethod.PHENE_DENSITY_FREQ],
    'fit': ['niching', 'novelty', 'nsga2', 'nslc', 'raw'],
    'genformat': ['1'],
    'popsize': [8],
    'archive': [8],
    'initialgenotype': [None],
    'pmut': [0.7],
    'pxov': [0.2],
    'tournament': [5]
}

SETTINGS_TO_TEST_FRAMS_ISLANDS = {
    'opt': ['velocity', 'vertpos'],
    'max_numparts': [None],
    'max_numjoints': [20],
    'max_numneurons': [20],
    'max_numconnections': [None],
    'max_numgenochars': [20],
    'hof_size': [0, 10],
    'migration_interval': [1,5],
    'number_of_populations':[1,5],
    'genformat': ['1'],
    'popsize': [8],
    'initialgenotype': [None],
    'pmut': [0.7],
    'pxov': [0.2],
    'tournament': [5]
}

def test_run_experiment_numerical(params):
    # multiple criteria not supported here. If needed, use FramsticksEvolution.py

    experiment = ExperimentNumerical(
                                    hof_size=params['hof_size'],
                                    popsize=params['popsize'],
                                    save_only_best=True,)

    experiment.evolve(hof_savefile=None,
                      generations=GENERATIONS,
                      initialgenotype=params['initialgenotype'],
                      pmut=params['pmut'],
                      pxov=params['pxov'],
                      tournament_size=params['tournament'])


def test_run_experiment_numerical_islands(params):
    # multiple criteria not supported here. If needed, use FramsticksEvolution.py

    experiment = ExperimentNumericalIslands(hof_size=params['hof_size'],
                                    popsize=params['popsize'],
                                    save_only_best=True,
                                    migration_interval=params['migration_interval'],
                                    number_of_populations=params['number_of_populations'])


    experiment.evolve(hof_savefile=None,
                      generations=GENERATIONS,
                      initialgenotype=params['initialgenotype'],
                      pmut=params['pmut'],
                      pxov=params['pxov'],
                      tournament_size=params['tournament'])

def test_run_experiment_frams_niching(params):
    # multiple criteria not supported here. If needed, use FramsticksEvolution.py
    opt_criteria = params['opt'].split(",")
    framsLib = FramsticksLib(
        parsed_args.path, parsed_args.lib, parsed_args.sim)
    constrains = {"max_numparts": params['max_numparts'],
                  "max_numjoints": params['max_numjoints'],
                  "max_numneurons": params['max_numneurons'],
                  "max_numconnections": params['max_numconnections'],
                  "max_numgenochars": params['max_numgenochars'],
                  }
                  
    old_fitness_set_negative_to_zero = Individual.fitness_set_negative_to_zero # save a copy of the current value to restore later
    Individual.fitness_set_negative_to_zero = True # niching must have it set to True, see "-fitness_set_negative_to_zero" argument in experiment_abc.py

    experiment = ExperimentFramsNiching(frams_lib=framsLib,
                                        optimization_criteria=opt_criteria,
                                        hof_size=params['hof_size'],
                                        constraints=constrains,
                                        normalize=params['normalize'],
                                        dissim=params['dissim'],
                                        crowding_dissim=params['crowding_dissim'],
                                        fit=params['fit'],
                                        genformat=params['genformat'],
                                        popsize=params['popsize'],
                                        archive_size=params['archive'],
                                        save_only_best=True,
                                        knn_niching=5,
                                        knn_nslc=5)

    experiment.evolve(hof_savefile=None,
                      generations=GENERATIONS,
                      initialgenotype=params['initialgenotype'],
                      pmut=params['pmut'],
                      pxov=params['pxov'],
                      tournament_size=params['tournament'])
    
    Individual.fitness_set_negative_to_zero = old_fitness_set_negative_to_zero # restore original value

def test_run_experiment_frams_island(params):
    # multiple criteria not supported here. If needed, use FramsticksEvolution.py
    opt_criteria = params['opt'].split(",")
    framsLib = FramsticksLib(
        parsed_args.path, parsed_args.lib, parsed_args.sim)
    constrains = {"max_numparts": params['max_numparts'],
                  "max_numjoints": params['max_numjoints'],
                  "max_numneurons": params['max_numneurons'],
                  "max_numconnections": params['max_numconnections'],
                  "max_numgenochars": params['max_numgenochars'],
                  }

    experiment = ExperimentFramsIslands(frams_lib=framsLib,
                                        optimization_criteria=opt_criteria,
                                        hof_size=params['hof_size'],
                                        constraints=constrains,
                                        genformat=params['genformat'],
                                        popsize=params['popsize'],
                                        migration_interval=params['migration_interval'],
                                        number_of_populations=params['number_of_populations'],
                                        save_only_best=True)

    experiment.evolve(hof_savefile=None,
                      generations=GENERATIONS,
                      initialgenotype=params['initialgenotype'],
                      pmut=params['pmut'],
                      pxov=params['pxov'],
                      tournament_size=params['tournament'])


def parseArguments():
    parser = argparse.ArgumentParser(
        description='Run this program with "python -u %s" if you want to disable buffering of its output.' % sys.argv[0])
    parser.add_argument('-path', type=ensureDir, required=True,
                        help='Path to Framsticks CLI without trailing slash.')
    parser.add_argument('-lib', required=False,
                        help='Library name. If not given, "frams-objects.dll" or "frams-objects.so" is assumed depending on the platform.')
    parser.add_argument('-sim', required=False, default="eval-allcriteria.sim",
                        help="The name of the .sim file with settings for evaluation, mutation, crossover, and similarity estimation. If not given, \"eval-allcriteria.sim\" is assumed by default. Must be compatible with the \"standard-eval\" expdef. If you want to provide more files, separate them with a semicolon ';'.")

    return parser.parse_args()


def get_params_sets(settings):
    params_sets = []
    for k in settings.keys():
        temp_param_set = []
        for value in settings[k]:
            if params_sets:
                for exsiting_set in params_sets:
                    copy_of_set = exsiting_set.copy()
                    copy_of_set[k] = value
                    temp_param_set.append(copy_of_set)
            else:
                temp_param_set.append({k: value})
        params_sets = temp_param_set
    return params_sets


def cover_to_test(params, run_exp):
    run_exp(params)
    return 1


def run_tests():
    results = []

    print("TESTING NUMERICAL")
    params_sets = get_params_sets(SETTINGS_TO_TEST_NUMERIC)
    print(f"Starting executing {len(params_sets)} experiments")
    results.extend([cover_to_test(params, test_run_experiment_numerical) for params in params_sets])

    print("TESTING NUMERICAL ISLANDS")
    params_sets = get_params_sets(SETTINGS_TO_TEST_NUMERIC_ISLAND)
    print(f"Starting executing {len(params_sets)} experiments")
    results.extend([cover_to_test(params,test_run_experiment_numerical_islands) for params in params_sets])

    print("TESTING FRAMS NICHING")
    params_sets = get_params_sets(SETTINGS_TO_TEST_FRAMS_NICHING)
    print(f"Starting executing {len(params_sets)} experiments")
    results.extend([cover_to_test(params, test_run_experiment_frams_niching) for params in params_sets])

    print("TESTING FRAMS ISLANDS")
    params_sets = get_params_sets(SETTINGS_TO_TEST_FRAMS_ISLANDS)
    print(f"Starting executing {len(params_sets)} experiments")
    results.extend([cover_to_test(params,test_run_experiment_frams_island) for params in params_sets])

    print(f"Passed tests: {sum(results)} / {len(results)}")


if __name__ == "__main__":
    parsed_args = parseArguments()
    run_tests()
