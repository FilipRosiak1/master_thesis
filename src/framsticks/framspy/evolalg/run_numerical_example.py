import numpy as np
from .numerical_example.numerical_example import ExperimentNumerical
from .structures.individual import Individual


def main():
    parsed_args = ExperimentNumerical.get_args_for_parser().parse_args()
    Individual.fitness_set_negative_to_zero = parsed_args.fitness_set_negative_to_zero # setting the "static" field once
    print("Argument values:", ", ".join(
        ['%s=%s' % (arg, getattr(parsed_args, arg)) for arg in vars(parsed_args)]))

    initialgenotype = np.array([100, 100, 100, 100])
    experiment = ExperimentNumerical(
        hof_size=parsed_args.hof_size,
        popsize=parsed_args.popsize,
        save_only_best=parsed_args.save_only_best)

    hof, stats = experiment.evolve(hof_savefile=parsed_args.hof_savefile,
                                   generations=parsed_args.generations,
                                   initialgenotype=initialgenotype,
                                   pmut=parsed_args.pmut,
                                   pxov=parsed_args.pxov,
                                   tournament_size=parsed_args.tournament)
    print('Best individuals:')
    for ind in hof:
        print(ind.rawfitness, '\t<--\t', ind.genotype)


if __name__ == "__main__":
    main()
