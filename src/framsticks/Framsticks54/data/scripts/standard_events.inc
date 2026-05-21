// standard user events handling (user can place food manually)
//
// to be used by experiment definitions supporting "food" group.
// requires the addfood function (should be implemented in the experiment definition).

function queryEventNames()
{
	return ["Place food", "Drop food"];
}

function onUserEvent(type, point, vector)
{
	//Simulator.print("called event #"+type);
	var p = WorldMap.intersect(point, vector);
	if (p)
	{
		//	Simulator.print("intersection found:"+p[0]+","+p[1]+","+p[2]);
		var food = addfood();
		if (food)
		{
			var z = p[2];
			if (type == 1)
				z += 10;
			food.locationSetBboxLow(p[0] - food.bboxSize.x / 2, p[1] - food.bboxSize.y / 2, z + food.bboxGroundOffset + 0.5);
		}
		// else: food could not be added
	}
	//else
	//	Simulator.print("no intersection...");
}
