// used in 'foodfinding-usercode.inc'

// assumes: start=(x=0,y=0), food=(x=1,y=0)
function foodfindingReward(x, y)
{
	var r = Math.sqrt(Math.pow(x - 1.0, 2) + Math.pow(y, 2));
	if (r <= 1.0)
		return 1.0 - r; //inside cone

	r = Math.sqrt(Math.pow(x, 2) + Math.pow(y, 2));
	if (r >= 2.0)
		return 0.0; //outside outer circle

	var max_reward = 1.0 - Math.abs(1.0 - r);
	var total_reward = max_reward * max_reward; //simplified formula, TODO discover the proper formula (compare with python sampling code)
	var a = Math.atan2(y, x);
	var edge = Math.acos(0.5 * r); //angle of the intersection of two circles
	var full = Math.pi - edge; //angle between (-x,0) and cone
	var distance = Math.abs(a) - edge; //0..edge
	// distance/full = 0..1 across the arc (cone edge to -X)
	var arc_len = full * r; //arc length of 'full'
	// total negative reward = max_negative * arc_len
	// max_negative = total/arc_len
	return -(total_reward / arc_len) * (distance / full); //TODO verify if properly scaled (compare with python sampling code)
}
