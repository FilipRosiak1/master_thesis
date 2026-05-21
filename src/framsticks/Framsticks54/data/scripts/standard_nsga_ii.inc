// NSGA-II
// TODO support custom performance fields from the Genotype.data dictionary, see getNsgaGoalDictionary()
// TODO the implementation below seems to ignore the genotype.instances field (which can be 0, 1, or more). This needs to be supported! For now, GROUP_IDENTICAL_GENOTYPES is disabled in standard.expdef because of this deficiency. 
// -------
// Convention of storing performance / fitness data in genotypes:
// Everything is stored in the Genotype.data["nsga"] field. The field type is a dictionary: data["nsga"] = {}.
// The "nsga" dictionary fields:
// int "rank" : current Pareto rank of this Genotype. Because fitness is maximized, Pareto ranks are encoded as:
//              0 is the highest rank (*the* Pareto front), -1 is the next rank (second Pareto front), and so on.
//              Rank is also used as Genotype.fitness for the purposes of (tournament) selection for creation.
// int "d_c" : domination count - used during evaluation of Pareto rank.
// Vector "S" : set of Genotypes dominated by this Genotype - used during the evaluation of the Pareto rank.
// float "c_dist" : crowding distance among other Genotypes in the same rank.
// string "attr" : attribute name used temporarily for determining c_dist.
// 
// This inc file uses a global variable defined in standard.expdef:
// extern g_update_nsga_ranks_needed;

function nsgaFitnessFormula()
{
	return "return this.data.hasKey(\"nsga\") ? this.data->nsga->rank : -999999;";
}

// Returns the worst member of the pool by the means of NSGA-II.
// Such worst element is in the furthest Pareto front, and if such front has multiple members - the one with lowest
// crowding distance is chosen.
// pool : The pool for the genotype to be selected from
// returns : Genotype
function selectNsgaWorst(pool)
{
	updateNsgaRanksIfNeeded(pool, null);
	var candidate = pool[0];
	for (var other in pool)
	{
		if (other.data->nsga->rank < candidate.data->nsga->rank ||
		        (other.data->nsga->rank == candidate.data->nsga->rank && other.data->nsga->c_dist < candidate.data->nsga->c_dist))
		{
			candidate = other;
		}
	}
	return candidate;
}

function attrCompare(a, b)
{
	return a.["" + a.data->nsga->attr] < b.["" + b.data->nsga->attr];
}

// Creates a dictionary with goal weights based on current experiment settings.
function getNsgaGoalDictionary()
{
	var criteria_k_v =
	{
		"cr_life": "lifespan",
		"cr_v": "velocity",
		"cr_gl": "numparts",
		"cr_joints": "numjoints",
		"cr_nnsiz": "numneurons",
		"cr_nncon": "numconnections",
		"cr_di": "distance",
		"cr_vpos": "vertpos",
		"cr_vvel": "vertvel"
	};
	var goal_dict = {};
	for (var i = 0; i < criteria_k_v.size; i++)
	{
		var prop = criteria_k_v.getKey(i), val = criteria_k_v.get(i);
		var weight = ExpProperties.[prop];
		if (weight != 0.0)
		{
			goal_dict[val] = weight;
		}
	}
	return goal_dict;
}

function updateNsgaRanksIfNeeded(pool, goal_dictionary)
{
	if (ExpProperties.cr_nsga && ExpState.update_nsga_ranks_needed) //"dirty" flag set?
	{
		updateNsgaRanks(pool, goal_dictionary);
		ExpState.update_nsga_ranks_needed = 0;
	}
}

// Calculates the new rank for this genepool's genotypes.
// To save unnecessary computation, this function should not be called directly; call updateNsgaRanksIfNeeded() instead.
// pool : genepool to evaluate fitness for.
// goal_dictionary : dictionary object where keys represent members of Genotype that represent the performance to
//                   measure, and weight is either positive (maximize) or negative (minimize).
function updateNsgaRanks(pool, goal_dictionary)
{
	//Simulator.print("updateNsgaRanks(), simstep=" + Simulator.stepNumber);
	if (pool == null) //second argument introduced mostly for debugging or additional flexibility?
	{
		pool = goal_dictionary;
		goal_dictionary = null;
	}
	if (goal_dictionary == null)
	{
		goal_dictionary = getNsgaGoalDictionary();
	}
	for (var g in pool)
	{
		if (!g.data.hasKey("nsga"))
		{
			g.data->nsga =
			{
				"d_c": 0,
				"S": [],
				"c_dist": 0.0,
				"attr": ""
			};
		}
		else
		{
			g.data->nsga->d_c = 0;
			g.data->nsga->S.clear();
			g.data->nsga->c_dist = 0.0;
		}
	}
	// Pair-wise comparison of dominance
	for (var i = 0; i < pool.size; i++)
	{
		var g_a = pool[i];
		for (var j = 0; j < pool.size; j++)
		{
			var g_b = pool[j];
			var a_is_better = 0, b_is_better = 0;

			for (var k = 0; k < goal_dictionary.size; k++)
			{
				var criteria = goal_dictionary.getKey(k), mod = goal_dictionary.get(k);

				var a_eval = mod * g_a.["" + criteria];
				var b_eval = mod * g_b.["" + criteria];
				if (a_eval > b_eval)
				{
					a_is_better++;
				}
				if (b_eval > a_eval)
				{
					b_is_better++;
					break;
				}
			}
			if (a_is_better > 0 && b_is_better == 0)
			{
				g_a.data->nsga->S.add(g_b);
				g_b.data->nsga->d_c++;
			}
		}
	}

	// Assigning ranks
	var current_front = [];
	var next_front = [];
	var rank = 0;
	for (var g in pool)
	{
		if (g.data->nsga->d_c == 0)
		{
			current_front.add(g);
		}
	}
	const BIGVAL = 100000000.0;
	while (current_front.size > 0)
	{
		// Calculate crowding distance
		for (var k = 0; k < goal_dictionary.size; k++)
		{
			var criteria = goal_dictionary.getKey(k);
			for (var g in current_front)
			{
				g.data->nsga->attr = criteria;
			}
			current_front.sort(function attrCompare);
			var span = current_front[current_front.size - 1].["" + criteria] - current_front[0].["" + criteria];
			if (span > 0)
			{
				current_front[0].data->nsga->c_dist += BIGVAL;
				current_front[current_front.size - 1].data->nsga->c_dist += BIGVAL;
				for (var i = 1; i < current_front.size - 1; ++i)
				{
					current_front[i].data->nsga->c_dist += (current_front[i + 1].["" + criteria] - current_front[i - 1].["" + criteria]) / span;
				}
			}
		}
		// Create new front
		for (var g in current_front)
		{
			g.data->nsga->rank = rank;
			//g.fit = rank;
			for (var inferior in g.data->nsga->S)
			{
				inferior.data->nsga->d_c--;
				if (inferior.data->nsga->d_c == 0)
				{
					next_front.add(inferior);
				}
			}
		}
		current_front = next_front;
		next_front = [];
		rank--;
	}
	pool.fitness = nsgaFitnessFormula(); //propagate the rank from internal Genotype dictionary to its fitness value
}
