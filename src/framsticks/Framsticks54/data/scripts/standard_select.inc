//standard selection methods

// new style: do not use global state (selected group index) in GenePools, return Genotype object

@include "standard_nsga_ii.inc"
function selectedForDeletionInPool(pool)
{
	if (ExpProperties._propertyExists("cr_nsga")) //in some expdefs, standard_props.inc that introduces cr_nsga is not included
		if (ExpProperties.cr_nsga)
			return selectNsgaWorst(pool); //dedicated negative selection method for NSGA-II
	if (ExpProperties.delrule == 0)
		return pool.random();
	if (ExpProperties.delrule == 1)
		return pool.revroulette();
	return pool.worst(); //"delrule" is 0..2
}

function selectedForCreationInPool(pool)
{
	if (ExpProperties._propertyExists("cr_nsga")) //in some expdefs, standard_props.inc that introduces cr_nsga is not included
		updateNsgaRanksIfNeeded(pool, null);
	if (ExpProperties.selrule == 0)
		return pool.random();
	if (ExpProperties.selrule == 1)
		return pool.roulette();
	return pool.tournament(ExpProperties.selrule);
}

function selectedSimilarInPool(pool, genotype)
{
	if (ExpProperties._propertyExists("cr_nsga")) //in some expdefs, standard_props.inc that introduces cr_nsga is not included
		updateNsgaRanksIfNeeded(pool, null);
	if (ExpProperties.selrule == 0)
		return pool.randomLikeGenotype(ExpProperties.xov_mins, genotype);
	else
		return pool.rouletteLikeGenotype(ExpProperties.xov_mins, genotype);
}

// below: deprecated, for compatiblity with old scripts - do not use: these functions will be eventually removed!

function selectedForDeletion()
{
	if (ExpProperties.delrule == 0)
		return GenePools.random();
	if (ExpProperties.delrule == 1)
		return GenePools.revroulette();
	return GenePools.worst();
}

function selectedForCreation()
{
	if (ExpProperties._propertyExists("cr_nsga")) //in some expdefs, standard_props.inc that introduces cr_nsga is not included
		updateNsgaRanksIfNeeded(GenePools[0], null);
	if (ExpProperties.selrule == 0)
		return GenePools.random();
	if (ExpProperties.selrule == 1)
		return GenePools.roulette();
	return GenePools.tournament(ExpProperties.selrule);
}

function selectedSimilar()
{
	if (ExpProperties._propertyExists("cr_nsga")) //in some expdefs, standard_props.inc that introduces cr_nsga is not included
		updateNsgaRanksIfNeeded(GenePools[0], null);
	if (ExpProperties.selrule == 0)
		return GenePools.randomLikeThis(ExpProperties.xov_mins);
	else
		return GenePools.likeThisRoulette(ExpProperties.xov_mins);
}
