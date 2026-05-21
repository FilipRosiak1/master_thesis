
//fileformat==0: separate creatures, and within each creature separate sections for parts and joints. No redundancy, but more difficult to parse.
//fileformat==1: creatures merged, just an array of part-to-part connections (i.e., joints), high redundancy, larger file, but easy to read and parse.
function dumpCreatures(direct, fileformat)
{
	var pop = Populations[0];
	var str="";	//the file may be read while being written here, so introduced this variable to shorten the file writing time and avoid writing incrementally
	if (fileformat==0)
	{
		str="# parts are points, joints are sticks.\n";
		str+="# format for parts: x y z vx vy vz\n";
		str+="# format for joints: part1index(0-based)  part2index(0-based)  stresslinear  stressrotational\n";
		for (var cr in pop)
		{
			str+="\n# creature_number  number_of_parts  number_of_joints\n";
			str+="%d %d %d\n" % cr.index % cr.numparts % cr.numjoints;
			for (var j = 0; j < cr.numparts; j++)
			{
				var mpart = cr.getMechPart(j);
				str+="%.3f %.3f %.3f %.4f %.4f %.4f\n" % mpart.x % mpart.y % mpart.z % mpart.vx % mpart.vy % mpart.vz;
			}
			for (var j = 0; j < cr.numjoints; j++)
			{
				var mjoint = cr.getMechJoint(j);
				str+="%d %d %g %g\n" % mjoint.joint.p1 % mjoint.joint.p2 % mjoint.stress % mjoint.rotstress;
			}
		}
	} else
	{
		str="# format: x1 y1 z1 x2 y2 z2 v stresslinear stressrotational\n"; 
		//f.writeComment(...);
		for (var cr in pop)
		{
			for (var j = 0; j < cr.numjoints; j++)
			{
				var mjoint = cr.getMechJoint(j);
				var mp1=cr.getMechPart(mjoint.joint.p1);
				var mp2=cr.getMechPart(mjoint.joint.p2);
				// calculate velocity of a stick as the average of velocities of its two parts
				var vx=(mp1.vx+mp2.vx)/2;
				var vy=(mp1.vy+mp2.vy)/2;
				var vz=(mp1.vz+mp2.vz)/2;
				var v=Math.sqrt(vx*vx+vy*vy+vz*vz);
				str+="%.3f %.3f %.3f %.3f %.3f %.3f %.4f %.4f %.4f\n" % mp1.x % mp1.y % mp1.z % mp2.x % mp2.y % mp2.z % v % mjoint.stress % mjoint.rotstress;
				//f.writeString(...);
			}
		}
	}

	var f;
	if (direct != 0)
		f = File.createDirect("dump.txt", "Coordinates, velocities and stress (linear and rotational).");
	else
		f = File.create("dump.txt", "Coordinates, velocities and stress (linear and rotational).");
	f.writeString(str);
	f.close();
}
