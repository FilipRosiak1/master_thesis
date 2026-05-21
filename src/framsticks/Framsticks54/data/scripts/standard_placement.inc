//standard placement function

function placeCreatureRandomly(cr, centered, rotation)
{
	var x, y, z;
	if (rotation > 0)
	{
		var angle;
		if (rotation == 4)
			angle = Math.rndUni(0, Math.twopi);
		else
			angle = [Math.pi, Math.pi2, Math.pi4][rotation - 1] * Math.random(8);
		cr.rotate(0, 0, angle);
	}
	if (centered == 0) //random x,y
	{
		x = (World.wrldsiz - cr.bboxSize.x) * Math.rnd01;
		y = (World.wrldsiz - cr.bboxSize.y) * Math.rnd01;
	}
	else //centered position
	{
		x = (World.wrldsiz - cr.bboxSize.x) / 2;
		y = (World.wrldsiz - cr.bboxSize.y) / 2;
	}
	z = WorldMap.getHeight(x, y);
	if (World.wrldwat > z) // water here!
	{
		if (ExpProperties.creath > 0)
			z = World.wrldwat + ExpProperties.creath;
		else
			z = z + (1.0 + ExpProperties.creath) * (Math.max(z, (World.wrldwat - cr.bboxSize.z + 1.0)) - z);
	}
	else // land area
	{
		if (ExpProperties.creath > 0)
			z += ExpProperties.creath;
	}

	cr.locationSetBboxLow(x, y, z + cr.bboxGroundOffset);
	//Simulator.message("placeCreatureRandomly() placed center of creature at %s" % cr.bboxCenter, 0);
	return ;
}

// for compatiblity with old scripts - do not use: this function will be eventually removed!
function place_randomly(cr, centered, rotation)
{
	placeCreatureRandomly(Creature.*, centered, rotation);
}
