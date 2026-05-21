function
ExpProperties_cr_c_set, ExpProperties_cr_life_set, ExpProperties_cr_v_set, ExpProperties_cr_gl_set,
ExpProperties_cr_joints_set, ExpProperties_cr_nnsiz_set, ExpProperties_cr_nncon_set, ExpProperties_cr_di_set,
ExpProperties_cr_vpos_set, ExpProperties_cr_vvel_set, ExpProperties_cr_norm_set, ExpProperties_cr_simi_set,
update_fitformula()
{
	var formula = "" + ExpProperties.cr_c;
	formula += singlecrit("cr_life", "lifespan");
	formula += singlecrit("cr_v", "velocity");
	formula += singlecrit("cr_gl", "numparts");
	formula += singlecrit("cr_joints", "numjoints");
	formula += singlecrit("cr_nnsiz", "numneurons");
	formula += singlecrit("cr_nncon", "numconnections");
	formula += singlecrit("cr_di", "distance");
	formula += singlecrit("cr_vpos", "vertpos");
	formula += singlecrit("cr_vvel", "vertvel");
	if (ExpProperties.cr_simi)
		formula = "(" + formula + ")*this.simi";
	GenePools[0].fitness = "return " + formula + ";";
	//Simulator.print("update_fitformula() built fitness: "+formula);
	ExpState.update_nsga_ranks_needed = 1;
}

function ExpProperties_cr_nsga_set()
{
	ExpState.update_nsga_ranks_needed = 1;
}

function singlecrit(crname, fieldname)
{
	var weight = ExpProperties.[crname];
	if (weight == 0.0)
		return "";
	if (ExpProperties.cr_norm)
		return "+this.getNormalized(Genotype:" + fieldname + ")*" + weight;
	else
		return "+this." + fieldname + "*" + weight;
}
