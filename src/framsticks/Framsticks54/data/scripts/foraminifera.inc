
function create_genotype(proculus_size, number_of_chambers, rgbstring, lastchambergrowth) //lastchambergrowth is 0..1
{
	const shift = 0.7;
	const angle_delta = 0.8;
	const angle_delta_delta = -0.01;
	const growing = 1.07; //7% growth

	var str = "//0s\nm:Vstyle=foram\n";
	var size = proculus_size;
	for(var i = 0; i < number_of_chambers; i++)
	{
		var effectivesize = size; //'effectivesize' is introduced only to consider the last chamber
		if (i == number_of_chambers - 1) //last chamber
		{
			effectivesize *= lastchambergrowth;
			size = size * (1.35 - 0.35 * lastchambergrowth); //last iteration: 'size' is only used for shifting (dx). The last chamber emerges at the surface of the previous one
			if (lastchambergrowth < 1)
				rgbstring = "0.9,0.9,0.9,i=\"growing=%g\"" % lastchambergrowth; //when the last chamber is growing, make it bright gray and add extra information in its "i" field
		}
		effectivesize = Math.max(effectivesize, MIN_PART_SIZE);
		str += "p:sh=1,sx=%g,sy=%g,sz=%g,rz=3.14159265358979,vr=%s\n" % effectivesize % effectivesize % effectivesize % rgbstring;
		if (i > 0)
			str += "j:%d,%d,sh=1,dx=%g,rz=%g\n" % (i - 1) % i % (size * shift) % (angle_delta + i * angle_delta_delta);
		size *= growing;
	}
	return str;
}

function setGenotype(mode)
{
	if (mode->opt == "growth")
	{
		mode->cr.data->genes = mode->parent_genes;
		mode->cr.data->lifeparams = mode->parent_lifeparams;
	}

	else if (mode->opt == "birth")
	{
		foram_uid += 1;
		var new_id = "c" + string(foram_uid);
		mode->cr.data->genes = String.deserialize(String.serialize(mode->genes));
		mode->cr.data->lifeparams = {"max_energy_level" : mode->energy0, "gen" : mode->gen,  "hibernated" : 0, "species" : mode->species, "reproduce" : 0, "dir" : randomDir(), "dir_counter" : Math.random(int(secToSimSteps(ExpProperties.dir_change_sec))), "chamber_growth" : -1, "division_time" : -1, "uid" : new_id};

		var oper = "cloning";
		var inherit = [1.0];
		if (mode->parentsuids.size > 1)
		{
			oper = "cross-over";
			inherit = [0.5, 0.5];
		}

		var dict = {"Time": Simulator.stepNumber, "FromIDs": mode->parentsuids, "ID": new_id, "Inherited": inherit, "Operation": oper, "Kind" : mode->gen};
		if (ExpProperties.print_evol_progress == 1) 
			Simulator.print("[OFFSPRING] " + String.serialize(dict));
	}
}

function getEnergy0(radius)
{
	return energyFromVolume(micronsToFrams(radius), 1);
}

function gametsDivision(parent_energy, energy0)
{
	var number = 1;
	var result = parent_energy;
	while ((result - ExpProperties.divisionCost) >= energy0)
	{
		result = (result - ExpProperties.divisionCost) / 2;
		number *= 2;
	}
	//Simulator.print("parent: " + parent_energy + " result: " + result + " number " + number);
	return {"energy" : result, "number" : number};
}

function reproduce_haploid(parent, parent2, clone)
{
	var number, energy0, new_genes, gen;
	if (clone == 1)
	{
		var offspring = gametsDivision(parent.energy, getEnergy0(getGene(parent, "energies0", 0)[0]));
		energy0 = offspring->energy;
		number = offspring->number;
		new_genes = parent.data->genes;
		parent.data->lifeparams->gen = 1 - parent.data->lifeparams->gen; //because of reversal of "gen" in createOffspring function
		gen = parent.data->lifeparams->gen;
	}
	else
	{
		var offspring1 = gametsDivision(parent.energy, getEnergy0(getGene(parent, "energies0", 0)[1]));
		var offspring2 = gametsDivision(parent2.energy, getEnergy0(getGene(parent2, "energies0", 0)[1]));
		energy0 = (offspring1->energy + offspring2->energy);
		number = ExpProperties.gametSuccessRate * (offspring1->number + offspring2->number) / 2;
		new_genes = [parent.data->genes, parent2.data->genes];
		gen = 1 - parent.data->lifeparams->gen;

		if (ExpProperties.logging == 1)
		{
			log(createLogVector(parent, parent.energy), ExpProperties.logPref + "repro_energies_log.txt");
			log(createLogVector(parent2, parent2.energy), ExpProperties.logPref + "repro_energies_log.txt");
			log(createLogVector(parent, number), ExpProperties.logPref + "repro_num_log.txt");
			log(createLogVector(parent, parent.lifespan), ExpProperties.logPref + "lifespan_log.txt");
			log(createLogVector(parent2, parent2.lifespan), ExpProperties.logPref + "lifespan_log.txt");
		}
	}

	//Simulator.print("haploid number of offspring: " + number + " energ0: " + energy0);

	for (var j = 0; j < number; j++)
	{
		createOffspring(create_genotype(ExpProperties.chamber_proculus_diplo, 1, colors[1], 1), energy0, new_genes, parent.data->lifeparams, [parent.data->lifeparams->uid, parent2.data->lifeparams->uid]);
	}
}

function reproduce_diploid(parent)
{
	var offspring = gametsDivision(parent.energy, getEnergy0(getGene(parent, "energies0", 0)[0]));
	var energy0 = offspring->energy;
	var number = offspring->number;

	if (ExpProperties.logging == 1)
	{
		log(createLogVector(parent, parent.energy), ExpProperties.logPref + "repro_energies_log.txt");
		log(createLogVector(parent, number), ExpProperties.logPref + "repro_num_log.txt");
		log(createLogVector(parent, parent.lifespan), ExpProperties.logPref + "lifespan_log.txt");
	}

	//Simulator.print("diploid number of offspring: " + number+ " energ0: " + energy0);

	for (var j = 0; j < number / 2; j++)
	{
		var crossed = 0;
		//crossover
		if (Math.rnd01 < ExpProperties.crossprob)
		{
			crossover(parent, "min_repro_energies");
			crossed = 1;
		}

		for (var k = 0; k < 2; k++)
		{
			createOffspring(create_genotype(ExpProperties.chamber_proculus_haplo, 1, colors[0], 1), energy0, parent.data->genes[0], parent.data->lifeparams, [parent.data->lifeparams->uid]);
		}

		//reverse of crossover for fossilization
		if (crossed == 1)
		{
			crossover(parent, "min_repro_energies");
			crossed = 0;
		}

	}
}

function reproduce_parents(species)
{
	var parent1 = null;
	var parent2 = null;
	var pop = Populations[0];
	for (var i = pop.size - 1; i >= 0; i--)
	{
		if (pop[i].data->lifeparams->reproduce == 1 && pop[i].data->lifeparams->species == species)
		{
			if ((pop[i].data->lifeparams->gen == 1) || ((pop[i].data->lifeparams->gen == 0) && ExpProperties.stress == 0))
			{
				continue;
			}
			else if (parent1 == null)
			{
				parent1 = pop[i];
			}
			else if (parent2 == null)
			{
				parent2 = pop[i];
			}
			if (parent1 != null && parent2 != null)
			{
				//when parents are ready for reproduction start gametogenesis
				if (parent1.data->lifeparams->division_time == -1 && parent2.data->lifeparams->division_time == -1)
				{
					var time = int(secToSimSteps(ExpProperties.gametoPeriodSec));
					parent1.data->lifeparams->division_time = time;
					parent2.data->lifeparams->division_time = time;
					parent1.idleen = 0;
					parent2.idleen = 0;
					//Simulator.print("parents "+parent1.uid + " " + parent2.uid + " ready to repro: "+Simulator.stepNumber);
				}
				//when gametogenesis is finished fuse gamets
				else if (parent1.data->lifeparams->division_time == 0 && parent2.data->lifeparams->division_time == 0)
				{
					reproduce_haploid(parent1, parent2, 0);
					//print_repro_info(parent1);
					//print_repro_info(parent2);
					pop.kill(parent1);
					pop.kill(parent2);
					parent1 = null;
					parent2 = null;
				}
			}
		}
	}
}

function readyToRepro(cr)
{
	var reproduced = 1;

	if (cr.data->lifeparams->gen == 1)
	{
		reproduce_diploid(cr);
	}

	else if (ExpProperties.stress == 0)
	{
		reproduce_haploid(cr, null, 1);
	}

	else
	{
		if (cr.signals.size == 0)
		{
			cr.signals.add("repro" + cr.data->lifeparams->species);
			cr.signals[0].power = 1;
		}
		reproduced = 0;
		cr.data->lifeparams->reproduce = 1;
	}

	if (reproduced == 1)
	{
		//print_repro_info(cr);
		Populations[0].kill(cr);
	}

	return reproduced;
}

function foramReproduce(cr)
{
	var properEnergy = cr.energy >= getGene(cr, "min_repro_energies", 0)[cr.data->lifeparams->gen];
	var reproduced = 0;

	//if creature has proper energy
	if ( properEnergy )
	{
		//reproduce with probability repro_prob
		if (Math.rnd01 <= ExpProperties.repro_prob) //TODO env trigger
		{
			reproduced = readyToRepro(cr);
		}
		else if (cr.signals.receive("repro" + cr.data->lifeparams->species) > 0)
		{
			reproduced = readyToRepro(cr);
		}
		if (reproduced == 1)
			return 1;
	}

	else if (!properEnergy)
	{
		cr.signals.clear();
		cr.data->lifeparams->reproduce = 0;
	}

	return 0;
}

function crossover(parent, gene)
{
	var tmp = parent.data->genes[0][gene];
	parent.data->genes[0][gene] = parent.data->genes[1][gene];
	parent.data->genes[1][gene] = tmp;
}

function createOffspring(geno, energy, parent_genes, parent_lifeparams, parentsuids)
{
	curColor = colors[1 - parent_lifeparams->gen];
	var cr = createAndRotate(geno, 0, 2 * Math.pi, 0);
	cr.energy0 = energy;
	cr.energy = cr.energy0;
	setGenotype({"opt" : "birth", "cr" : cr, "gen" : 1 - parent_lifeparams->gen, "species" : parent_lifeparams->species, "energy0" : cr.energy0, "genes" : parent_genes, "parentsuids" : parentsuids});
	placeRandomlyNotColliding(cr);
}

function print_repro_info(cr)
{
	if (ExpProperties.print_evol_progress == 1)
		Simulator.print("Reproduced " + cr.data->lifeparams->gen + " of species " + cr.data->lifeparams->species + " energy: " + cr.energy);
}
