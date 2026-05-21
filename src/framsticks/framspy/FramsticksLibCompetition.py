import sys
from time import perf_counter, strftime
from typing import List  # to be able to specify a type hint of list(something)
import numpy as np
from FramsticksLib import FramsticksLib
from base64 import urlsafe_b64encode


class FramsticksLibCompetition(FramsticksLib):
	"""A proxy to FramsticksLib.py with the same interface, but recording the highest achieved fitness and limiting the number of evaluation calls.
	Use it in the same way as FramsticksLib.py.
	For the use in competition, remember to call end() when your algorithm completes.
	To run a working example, follow these four steps:
	- set STORE_ALL_PART_COORDS = 0 in recording-body-coords.sim,
	- set SIMPLE_FITNESS_FORMAT = False below,
	- edit FramsticksEvolution.py in a few places so that it only uses the FramsticksLibCompetition class instead of FramsticksLib,
	- run: python FramsticksEvolution.py -path %DIR_WITH_FRAMS_LIBRARY%  -sim "eval-allcriteria.sim;deterministic.sim;recording-body-coords.sim"  -opt COGpath  -generations 20
	See also: https://www.framsticks.com/gecco-competition
	"""

	COMPETITOR_ID = 'AliceTeam'
	SIMPLE_FITNESS_FORMAT = True  # set to False only if you want compatibility with existing sources of optimization algorithms such as FramsticksEvolution.py. Otherwise (for True), you will just get a simple number as fitness.
	FITNESS_DICT_KEY = 'COGpath'  # only used for SIMPLE_FITNESS_FORMAT = False

	MAX_EVALUATIONS = 100_000  # 100k
	MAX_TIME = 60 * 60 * 1  # 1h (excluding evaluation time)

	TEST_FUNCTION = 3

	_best_fitness = None
	_best_solution = None
	_evaluation_count = 0
	_evaluation_time = 0  # used to exclude solution evaluation time from the total running time
	_time0 = perf_counter()


	def _evaluate_path(self, path):
		path = np.array(path)
		if self.TEST_FUNCTION == 3:
			return np.linalg.norm(path[0] - path[-1])  # simple example: returns distance between COG locations of birth and death.
		elif self.TEST_FUNCTION == 4:
			return np.linalg.norm(path[0] - path[-1]) * np.mean(np.maximum(0, path[:, 2]))  # simple example: run far and have COG high above ground!
		elif self.TEST_FUNCTION == 5:
			return 1000 - np.linalg.norm(np.linspace(0, 10, len(path), endpoint=True) - path[:, 2]) / np.sqrt(len(path))  # simple example: z coordinate of the COG should grow linearly from 0 to 1 during lifespan. Returns RMSE as a deviation measure (negative because we are maximizing, and offset to ensure positive outcomes so there is no clash with other optimization code that may assume that negative fitness indicates an invalid genotype).
		raise RuntimeError('TEST_FUNCTION==%s not implemented!' % self.TEST_FUNCTION)


	def _evaluate_single_genotype(self, genotype):
		self._evaluation_count += 1
		if self._evaluation_count > self.MAX_EVALUATIONS or perf_counter() - self._time0 - self._evaluation_time > self.MAX_TIME:
			print('The allowed time or the number of evaluations exceeded')
			self.end()  # exits the program
		result = super().evaluate([genotype])  # sample result for invalid genotype: [{'num': 172, 'name': 'Agoha Syhy', 'evaluations': None}]
		valid_result = result[0]['evaluations'] is not None
		fitness = self._evaluate_path(result[0]['evaluations']['']['data->bodyrecording']) if valid_result else None
		if fitness is not None and (self._best_fitness is None or self._best_fitness < fitness):
			self._best_fitness = fitness
			self._best_solution = genotype
		if self.SIMPLE_FITNESS_FORMAT:
			return fitness
		else:  # update existing structure (vector of dict of dict...)
			if valid_result:
				result[0]['evaluations'][''][self.FITNESS_DICT_KEY] = fitness
			else:
				# result[0]['evaluations'] = {'': {self.FITNESS_KEY: fitness}}  # [{'num': 260, 'name': 'Imepak Syhy', 'evaluations': {'': {'path': None}}}]
				pass  # leave 'result' as it is, the caller expects such an incomplete structure (with 'evaluations': None) on evaluation failure
			return result[0]


	def evaluate(self, genotype_list: List[str]):
		"""
		:return: a list of fitness values (see also SIMPLE_FITNESS_FORMAT), with None for genotypes that are not valid.
		"""
		if len(genotype_list) > self.MAX_EVALUATIONS:
			raise RuntimeError('Too many genotypes to evaluate in one batch: %d' % len(genotype_list))
		eval_time0 = perf_counter()
		fitnesses = [self._evaluate_single_genotype(genotype) for genotype in genotype_list]
		self._evaluation_time += perf_counter() - eval_time0
		return fitnesses


	def end(self):
		print('Finishing... best solution =', self._best_fitness)

		filename = urlsafe_b64encode(self.COMPETITOR_ID.encode()).decode() + ".results"
		competitor = "".join(x for x in self.COMPETITOR_ID if x.isalnum())

		s = strftime("%Y-%m-%d %H:%M")
		s += "\t%s\t%d\t%d" % (competitor, self.TEST_FUNCTION, self._evaluation_count)

		total_running_time = perf_counter() - self._time0
		s += "\t%g\t%g" % (total_running_time, total_running_time - self._evaluation_time)

		s += "\t" + str(self._best_fitness)
		s += "\t" + str(self._best_solution)
		print(s)
		with open(filename, "a") as outfile:  # append
			outfile.write(s)
			outfile.write("\n")
		print("Saved '%s' (%s)" % (filename, competitor))
		sys.exit()  # only call end() once
