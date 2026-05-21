function addTheaterObject()
{
        var o = Obj3D.load("theater");
        o["swiatlol"].transparent=1;
        o["swiatlop"].transparent=1;
	var e = Obj3D.newEmpty();
	e.add(o).transform = Matrix.newScale(0.3);
	GLDisplay.world3d.add(e);
	return e;
}

function createGlassMaterial()
{
	var mat = Material.new();
	mat.color = 0x444488cc;
	mat.shader = "glassreflection_env";
	return mat;
}

function addBottleObject()
{
	var o = Obj3D.newEmpty();
        var demoworld = Obj3D.load("demoworld");
	o.add(demoworld);
	demoworld.transform = Matrix.newTranslation((0, 0, -0.4));
	o.add(Obj3D.load("coverlabel")).transform = Matrix.newTranslation((0, 0, -0.4));
	var cover;
	cover = Obj3D.load("coverhole");
	o.add(cover).transform = Matrix.newTranslation((0, 0, -0.4));
	cover.transparent = 1;
	cover.material = createGlassMaterial();

	GLDisplay.world3d.add(o);
  	GLDisplay.world3d.filter = GLDisplay.world3d.filter | 1024;
  	demoworld.filter = demoworld.filter | 1024; //visible in reflections
	o.filter = o.filter | 1024; //visible in reflections

	return o;
}
