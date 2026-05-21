global g_clickable;

function addClickable(pos, size, id)
{
	if (g_clickable == null) g_clickable = [];
	var c = [pos, size, id, GUI.newButton(id, "")];
	c[3].userdata = c;
	c[3].style = 1;
	g_clickable.add(c);
}

function handleClickable()
{
	if (g_clickable == null) return;
	for (var c in g_clickable)
	{
		if (!c[3].enabled) continue;
		var pos = GLDisplay.project3D(GLDisplay.world3d, c[0]);
		var tmp = c[0].clone();
		tmp.x += c[1];
		var pos2 = GLDisplay.project3D(GLDisplay.world3d, tmp);
		tmp.x -= c[1];
		tmp.y += c[1];
		var pos3 = GLDisplay.project3D(GLDisplay.world3d, tmp);
		if (pos && pos2 && pos3)
		{
			pos2.sub(pos);
			pos3.sub(pos);
			var size = (pos2.length + pos3.length) / 2;
			c[3].setRect(pos.x - size / 2, pos.y - size / 2, size, size);
			c[3].visible = 1;
		}
		else
			c[3].visible = 0;
	}
}
