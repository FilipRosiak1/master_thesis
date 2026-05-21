// This file implements standardExpLoad() and standardExpSave() - the standard high-level load/save functions intended for files with *.expt extension.
// The most basic way of calling both functions is using the predefined constants, usually LOADSAVE_EVERYTHING_BUT_CREATURES or LOADSAVE_EVERYTHING.
// LOADSAVE_* constants are only sufficient if you want to allow or restrict the entire class of objects. For more detailed control, like saving
// only specific instances of populations or pools, the argument can be the object array.
// Examples:
//  - LOADSAVE_EVERYTHING_BUT_CREATURES - typical behavior of standard.expdef
//  - LOADSAVE_EVERYTHING - also saves Creature objects (beware: their state is not fully restored, breaking the usual performance calculation flow. Only use it if you are sure you accept the consequences)
//  - LOADSAVE_SIMPARAMS - only simulator parameters (including ExpProperties and ExpState)
//  - [sim_params.*] - same as above, using the array syntax of the argument
//  - LOADSAVE_SIMPARAMS | LOADSAVE_POPULATIONS - simulator parameters + population settings
//  - [sim_params.*,Populations[0],Populations[2]] - sim parameters + genepool parameters of groups 0 and 2 (and no genotypes)
//  - [sim_params.*,GenePools[0],LOADSAVE_GENOTYPES] - sim parameters + first genepool parameters + genotypes of the first pool
//  - [sim_params.*,GenePools[0],LOADSAVE_GENOTYPES,Populations[1],LOADSAVE_CREATURES] - sim parameters + first genepool (incl. genotypes) + second population (incl. creatures)
//  - [sim_params.*,LOADSAVE_SKIPOBJECT,Populations[1],LOADSAVE_CREATURES] - sim parameters + second population creatures, but skipping the population settings (the Population object must be listed in order to select which Creatures are to be saved/loaded; LOADSAVE_SKIPOBJECT skips the Population object while saving). Note that skipping GenePool or Population objects while saving may cause Genotypes and Creatures, when loaded, to go into different GenePools or Populations than they were originally in.
// Note: Excluding sim_params.* (or LOADSAVE_SIMPARAMS) produces an expt file which does not contain the experiment name (normally saved in the Simulator.expdef field), thus breaking the expdef-switching logic during loading.

const LOADSAVE_SIMPARAMS = 1;
const LOADSAVE_GENEPOOLS = 2;
const LOADSAVE_GENOTYPES = 4;
const LOADSAVE_POPULATIONS = 8;
const LOADSAVE_CREATURES = 16;
const LOADSAVE_EVERYTHING_BUT_CREATURES = LOADSAVE_SIMPARAMS | LOADSAVE_GENEPOOLS | LOADSAVE_GENOTYPES | LOADSAVE_POPULATIONS;
const LOADSAVE_EVERYTHING = LOADSAVE_EVERYTHING_BUT_CREATURES | LOADSAVE_CREATURES;
const LOADSAVE_SKIPOBJECT = 32;

global expLoadArgs, expLoadGenotypesInto, expLoadCreaturesInto;

function findObjectOfClass(arr, name)
{
	for(var i = 0; i < arr.size; i++) if (typeof(arr[i]) == name) return i;
	return -1;
}

function countObjectsOfClass(arr, name)
{
	var count = 0;
	for(var i = 0; i < arr.size; i++)
		if (typeof(arr[i]) == name)
			count++;
	return count;
}

function standardExpLoad(what)
{
	expLoadArgs = ExpLoadSave_TransformArgs(what);
	expLoadGenotypesInto = null;
	expLoadCreaturesInto = null;
	var expected_gpools = countObjectsOfClass(expLoadArgs, "GenePool");
	var expected_pops = countObjectsOfClass(expLoadArgs, "Population");

	for (var pool in GenePools)
		pool.clear();
	for (var pop in Populations)
		pop.clear();

	if (findObjectOfClass(expLoadArgs, "sim_params") >= 0)
		Loader.addClass(sim_params.*);
	if (findObjectOfClass(expLoadArgs, "Population") >= 0)
		Loader.addClass(Population.*);
	if (findObjectOfClass(expLoadArgs, "GenePool") >= 0)
		Loader.addClass(GenePool.*);

//  Loader.setBreakLabel(Loader.OnComment, "onExpLoad_Comment");
	Loader.setBreakLabel(Loader.BeforeObject, "onExpLoad_Object");
	Loader.setBreakLabel(Loader.BeforeUnknown, "onExpLoad_Unknown");
	Loader.run();

	var count = countObjectsOfClass(expLoadArgs, "GenePool");
	if (count > 0)
		Simulator.message("onExpLoad(): expected " + expected_gpools + " GenePool objects in the input file, found " + (expected_gpools - count), 1);
	count = countObjectsOfClass(expLoadArgs, "Population");
	if (count > 0)
		Simulator.message("onExpLoad(): expected " + expected_pops + " Population objects in the input file, found " + (expected_pops - count), 1);

	var loaded_gen = 0, loaded_cre = 0;
	for (var pool in GenePools)
		loaded_gen += pool.size;
	for (var pop in Populations)
		loaded_cre += pop.size;
	if ((loaded_gen > 0) || (loaded_cre > 0))
		Simulator.print("Experiment loaded (" + loaded_gen + " genotypes, " + loaded_cre + " creatures)");
        if (ExpState._propertyExists("update_nsga_ranks_needed"))
                ExpState.update_nsga_ranks_needed = 1;
}

// compare types first so we don't get warnings when comparing two objects of different types (classes)
function safeEquals(v1, v2)
{
	return (typeof(v1) == typeof(v2)) && (v1 == v2);
}

// return 1=>obj should be loaded, 0=>skip this obj, -1=>do nothing (already handled)
function loadOrSkipObject()
{
	var found;

	switch (Loader.objectName)
	{
	case "Population":
	case "GenePool":
		found = findObjectOfClass(expLoadArgs, Loader.objectName);
		if (found < 0) return 0;
		if (found < (expLoadArgs.size - 1))
		{
			if (safeEquals(expLoadArgs[found + 1], LOADSAVE_CREATURES))
			{
				expLoadCreaturesInto = expLoadArgs[found];
				expLoadArgs.remove(found + 1);
			}
			else if (safeEquals(expLoadArgs[found + 1], LOADSAVE_GENOTYPES))
			{
				expLoadGenotypesInto = expLoadArgs[found];
				expLoadArgs.remove(found + 1);
			}
		}
		Loader.currentObject = expLoadArgs[found];
		expLoadArgs.remove(found);
		if ((found > 0) && (safeEquals(expLoadArgs[found - 1], LOADSAVE_SKIPOBJECT)))
		{
			expLoadArgs.remove(found - 1);
			Simulator.message("onExpLoad(): unexpected " + Loader.objectName + " object in file", 2);
			return 0;
		}
		else
			return 1;
	}

	return -1;
}

function expLoadFindPool(objname, what, pooltype)
{
	var found = expLoadArgs.find(what);
	var ret = null;
	if ((found >= 0) && (typeof(expLoadArgs[found - 1]) == pooltype))
	{
		ret = expLoadArgs[found - 1];
		expLoadArgs.remove(found);
		expLoadArgs.remove(found - 1);
		if (((found - 2) >= 0) && (safeEquals(expLoadArgs[found - 2], LOADSAVE_SKIPOBJECT)))
			expLoadArgs.remove(found - 2);
		else
			Simulator.message("onExpLoad(): loading " + objname + "(s) without the preceding " + pooltype + " object", 1);
	}
	return ret;
}

function onExpLoad_Unknown()
{
	switch (Loader.objectName)
	{
	case "org":
		Loader.currentObject = Genotype.newFromString("");
		Interface.makeFrom(Loader.currentObject).setAllDefault();
		Loader.loadObject();
		if (expLoadGenotypesInto == null)
			expLoadGenotypesInto = expLoadFindPool("Genotype", LOADSAVE_GENOTYPES, "GenePool");
		if (expLoadGenotypesInto)
			Loader.currentObject.clone().moveTo(expLoadGenotypesInto);
		else
			Simulator.message("onExpLoad(): skipping unexpected Genotype object '" + Loader.currentObject.name + "'", 2);
		break;

	case "Creature":
		Loader.currentObject = CreatureSnapshot.new();
		Loader.loadObject();
		if (expLoadCreaturesInto == null)
			expLoadCreaturesInto = expLoadFindPool("Creature", LOADSAVE_CREATURES, "Population");
		if (expLoadCreaturesInto)
			expLoadCreaturesInto.add(Loader.currentObject);
		else
			Simulator.message("onExpLoad(): skipping unexpected Creature object '" + Loader.currentObject.name + "'", 2);
		break;

	case "CreaturesGroup":
		Loader.objectName = "Population";
		Simulator.message("'CreaturesGroup' found in experiment settings file, please change to 'Population'", 1);
		if (loadOrSkipObject() == 1) //for unknown the default action is to skip, so we force load when == 1
			Loader.loadObject();
		break;

	case "GenotypeGroup":
		Loader.objectName = "GenePool";
		Simulator.message("'GenotypeGroup' found in experiment settings file, please change to 'GenePool'", 1);
		if (loadOrSkipObject() == 1) //for unknown the default action is to skip, so we force load when == 1
			Loader.loadObject();
		break;
	}
}

function onExpLoad_Object()
{
	if (loadOrSkipObject() == 0) //for known object the default action is to load, so we force skip when == 0
		Loader.skipObject();
}

function ExpLoadSave_TransformArgs(what)
{
	var objs = [];
	if (typeof(what) == typeof(objs))
		objs = what;
	else
	{
		if (what & LOADSAVE_SIMPARAMS)
			objs.add(sim_params.*);
		if (what & LOADSAVE_GENEPOOLS)
			for (var gp in GenePools)
			{
				objs.add(gp);
				if (what & LOADSAVE_GENOTYPES)
					objs.add(LOADSAVE_GENOTYPES);
			}
		if (what & LOADSAVE_POPULATIONS)
			for (var pop in Populations)
			{
				objs.add(pop);
				if (what & LOADSAVE_CREATURES)
					objs.add(LOADSAVE_CREATURES);
			}
	}

	return objs;
}

function standardExpSave(what)
{
	var saved_genotypes = 0, saved_creatures = 0;
	var objs = ExpLoadSave_TransformArgs(what);

	File.writeComment("saved by '%s.expdef' using 'standard_exploadsave.inc'" % Simulator.expdef);

	var prev_o = null, skip = 0;
	for(var o in objs)
	{
		if (typeof(o) == "int")
		{
			if ((o == LOADSAVE_GENOTYPES) && (typeof(prev_o) == "GenePool"))
			{
				for (var g in prev_o)
				{
					File.writeNameObject("org", g);
					saved_genotypes++;
				}
			}
			else if ((o == LOADSAVE_CREATURES) && (typeof(prev_o) == "Population"))
			{
				for (var cr in prev_o)
				{
					File.writeObject(cr);
					saved_creatures++;
				}
			}
			else if (o == LOADSAVE_SKIPOBJECT)
				skip = 1;
		}
		else
		{
			if (!skip)
				File.writeObject(o);
			skip = 0;
		}
		prev_o = o;
	}

	Simulator.print("Experiment saved (" + saved_genotypes + " genotypes, " + saved_creatures + " creatures)");
}
