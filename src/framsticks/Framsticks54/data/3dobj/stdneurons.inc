function default_neuro_build()
{
  Element.useNeuro();
  if (n.class=="Light")
{
  GeomBuilder.addBranch(GeomBuilder.currentNode);
  var br=GeomBuilder.currentNode;
  Element.node=br;
  Loader.load("light-base.ac");
  if (Loader.loaded) GeomBuilder.add(Loader.loaded);
  GeomBuilder.addTransform(br);
  Loader.load("light-on.ac");
  if (Loader.loaded) GeomBuilder.add(Loader.loaded);
}
  else
{
  var filename;
  if (n.class == "|")
    filename = "neuro-bend.ac";
  else
    filename = "neuro-" + n.class + ".ac";
  Loader.load(filename);
  if (Loader.loaded)
    {
    GeomBuilder.addTransform(GeomBuilder.currentNode);
    GeomBuilder.matrixReset();
    if (n.liveNeuro)
      {
      if ((n.class == "Gpart") && (n.liveNeuro.mechpart.part.sh!=0))
        { //special case: "Gpart" in solids mode.
	  //scale by 5x its part.sx/sy/sz to make the ring encircle the part. (5x because the "Gpart" base model is designed to match the default stick width, so that the ball-and-stick mode does not have to be a special case.)
	  var part=n.liveNeuro.mechpart.part;
	  GeomBuilder.matrixScale(5*part.sx,5*part.sy,5*part.sz);
        }
      GeomBuilder.matrixMove1(n.liveNeuro.relative_pos);
      GeomBuilder.matrixRotate1(n.liveNeuro.relative_orient);
      GeomBuilder.setMatrix(GeomBuilder.currentNode);
      }
    GeomBuilder.add(Loader.loaded);
    }
 }
}

function default_neuro_update()
{
  Element.useNeuro();
  if (n.class=="Light")
	{
	var tr=GeomBuilder.getBranch(Element.node,1);
	if (tr)
		{
		GeomBuilder.matrixReset();
		var sc=n.state;
		if (sc<0.1) sc=0.1; else if (sc>1.0) sc=1.0;
		GeomBuilder.matrixScale(sc,sc,sc);
		GeomBuilder.setMatrix(tr);
		}
	}
}
