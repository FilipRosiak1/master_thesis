global g_quests, g_quests_by_id, g_quests_state, g_quest_categories, g_quest_categories_by_id;

function initQuestDefinitions()
{
	//layout: [ {id->"string", cat->"cat_id", crit->[ ["field","relation",value]*N ] (added automatically: goal->number)  }*N ]
	//"origin" field can only use "=" relation
	g_quests= [
		//f9
		{"id":"build9_dp7","cat":"bbf9","crit":[["format","=","9"],["origin","=","D"],["numparts",">=",7]]},
		{"id":"vert9_d5","cat":"bbf9","crit":[["format","=","9"],["origin","=","D"],["vertpos",">",0.5]]},
		{"id":"vert9_e5","cat":"bbf9","crit":[["format","=","9"],["origin","=","E"],["vertpos",">",0.5]]}, //this: origin=evolve a creature for the same criteria as the previous origin=design creature
		{"id":"vert9_d5j7","cat":"bbf9","crit":[["format","=","9"],["origin","=","D"],["vertpos",">",0.5],["numjoints","<=",7]]},
		{"id":"vert9_d15p8","cat":"bbf9","crit":[["format","=","9"],["vertpos",">",1.5],["numparts","<",8]]},
		//f1
		{"id":"build1_dp8","cat":"bbf1","crit":[["format","=","1"],["origin","=","D"],["numparts",">=",8]]},
		{"id":"vert1_15p10","cat":"bbf1","crit":[["format","=","1"],["numparts","<",10],["vertpos",">",1.5]]},
		{"id":"vert1_e2","cat":"bbf1","crit":[["format","=","1"],["origin","=","E"],["vertpos",">",2]]}, //this last goal from "vertpos" requires evolution, a segway into the following goals (for velocity) that all concern evolution
		//any format
		{"id":"speed_01p3","cat":"run","crit":[["numparts",">",3],["velocity",">",0.01]]},
		{"id":"speed_02","cat":"run","crit":[["velocity",">",0.02]]},
		{"id":"speed_02p7","cat":"run","crit":[["numparts",">",7],["velocity",">",0.02]]},
		{"id":"wspeed_001","cat":"swim","crit":[["water_velocity",">",0.01]]}, // >0.01 not easy to evolve, but we want it because it requires repetitive, periodic movement (not just a single muscle bending)
		//designing genetic look-alikes
		{"id":"simil_f9_square","cat":"similarity","crit":[["format","=","9"],["origin","=","D"],["numparts","=",4],["numneurons","=",0],["similar_to","<","""//0
p:-0.0035, 0.0005, -0.0025
p:-1.0025, 0.0035, 0.0045, vr=0.666666666666667, 0.5, 0.0
p:-1.0015, -0.0035, 1.0015, vr=0.166666666666667, 0.833333333333333, 0.5
p:-0.0005, -0.0005, 0.9985, vr=0.5, 0.333333333333333
j:0, 1, vg=0.0, 0.0
j:1, 2, vr=0.333333333333333, vb=0.0
j:2, 3, vr=0.0, 0.666666666666667
j:3, 0, vg=0.0
"""]]},
		{"id":"simil_f9_Tshape","cat":"similarity","crit":[["format","=","9"],["origin","=","D"],["numparts","=",8],["numneurons","=",0],["similar_to","<","""//0
p:-0.0035, 0.0005, -0.0025
p:-1.0025, 0.0035, 0.0045, vr=0.5, 0.333333333333333, 0.5
p:-1.0015, -0.0035, 1.0015, vr=0.944444444444444, 0.777777777777778, 0.0
p:-0.0005, -0.0005, 0.9985, vr=0.611111111111111, vb=0.0
p:0.0005, 0.0025, -1.0045, vr=0.0, vb=0.5
p:-0.9985, -0.0045, -0.9975, vr=0.0, 0.833333333333333, 0.888888888888889
p:-0.9975, -1.0015, -0.0005, vr=0.222222222222222, 0.0555555555555556
p:0.0035, -0.9985, -0.0035, vr=0.722222222222222, 0.0
j:0, 1, vg=0.0, 0.0
j:1, 2, vg=0.555555555555556, 0.0
j:2, 3, vr=0.888888888888889, vb=0.0
j:3, 0, vr=0.333333333333333, vb=0.0
j:0, 4, vr=0.0, vb=0.222222222222222
j:4, 5, vr=0.0, vb=0.777777777777778
j:5, 1, vr=0.0, 0.666666666666667
j:1, 6, vr=0.0, 0.111111111111111
j:6, 7, vr=0.444444444444445, 0.0
j:7, 0, vg=0.0
"""]]},
		{"id":"simil_f9_cube","cat":"similarity","crit":[["format","=","9"],["origin","=","D"],["numparts","=",8],["numneurons","=",0],["similar_to","<","""//0
p:-0.0035, 0.0005, -0.0025, vr=0.606060606060606, 0.666666666666667, 0.0
p:-1.0025, 0.0035, 0.0045, vr=0.696969696969697, 0.151515151515152, 0.333333333333333
p:-1.0015, -0.0035, 1.0015, vr=0.848484848484848, 0.454545454545455, 0.333333333333333
p:-0.0005, -0.0005, 0.9985
p:0.0005, 1.0025, -0.0045, vr=0.0606060606060606, 0.787878787878788, 0.424242424242424
p:-0.9985, 0.9955, 0.0025, vr=0.0303030303030303, 0.666666666666667, 0.666666666666667
p:-0.9975, 0.9985, 0.9995, vr=0.181818181818182, 0.606060606060606, 0.909090909090909
p:0.0035, 1.0015, 0.9965, vr=0.333333333333333, 0.393939393939394
j:0, 1, vg=0.0, 0.0
j:1, 2, vg=0.454545454545455, 0.0
j:2, 3, vg=0.909090909090909, 0.0
j:3, 0, vr=0.636363636363636, vb=0.0
j:0, 4, vr=0.181818181818182, vb=0.0
j:4, 5, vr=0.0, vb=0.272727272727272
j:5, 6, vr=0.0, vb=0.727272727272727
j:6, 7, vr=0.0, 0.818181818181818
j:7, 4, vr=0.0, 0.363636363636363
j:5, 1, vr=0.0909090909090908, 0.0
j:2, 6, vr=0.545454545454545, 0.0
j:7, 3, vg=0.0
"""]]},
		{"id":"simil_f1_Yshape","cat":"similarity","crit":[["format","=","1"],["origin","=","D"],["numparts","=",6],["numneurons","=",0],["similar_to","<","""//0
p:vr=0.5, 0.5, 0.5
p:1.0, vr=0.5, 0.5, 0.5
p:2.0, vr=0.5, 0.5, 0.5
p:3.0, vr=0.5, 0.5, 0.5
p:3.50017107459707, -0.865926611287823, vr=0.5, 0.5, 0.5
p:3.50017107459707, 0.865926611287823, vr=0.5, 0.5, 0.5
j:0, 1, dx=1.0, 0.0, 0.0, vr=0.5, 0.5, 0.5
j:1, 2, dx=1.0, 0.0, 0.0, vr=0.5, 0.5, 0.5
j:2, 3, dx=1.0, 0.0, 0.0, vr=0.5, 0.5, 0.5
j:3, 4, rz=-1.047, dx=1.0, 0.0, 0.0, vr=0.5, 0.5, 0.5
j:3, 5, rz=1.047, dx=1.0, 0.0, 0.0, vr=0.5, 0.5, 0.5
"""]]},
		{"id":"simil_f1_blender","cat":"similarity","crit":[["format","=","1"],["origin","=","D"],["numparts","=",7],["numneurons","=",0],["similar_to","<","""//0
p:vr=0.5, 0.5, 0.5
p:1.0, vr=0.5, 0.5, 0.5
p:1.50017107459707, -0.865926611287823, vr=0.5, 0.5, 0.5
p:1.50017107459707, 0.865926611287823, vr=0.5, 0.5, 0.5
p:2.00034214919414, 1.73185322257565, vr=0.5, 0.5, 0.5
p:2.25066046456068, 2.16487963488465, -0.865926594599594, vr=0.5, 0.5, 0.5
p:2.25036604155477, 2.1650496976467, 0.865926594599594, vr=0.5, 0.5, 0.5
j:0, 1, dx=1.0, 0.0, 0.0, vr=0.5, 0.5, 0.5
j:1, 2, rz=-1.047, dx=1.0, 0.0, 0.0, vr=0.5, 0.5, 0.5
j:1, 3, rz=1.047, dx=1.0, 0.0, 0.0, vr=0.5, 0.5, 0.5
j:3, 4, rx=1.5706, dx=1.0, 0.0, 0.0, vr=0.5, 0.5, 0.5
j:4, 5, rz=-1.047, dx=1.0, 0.0, 0.0, vr=0.5, 0.5, 0.5
j:4, 6, rz=1.047, dx=1.0, 0.0, 0.0, vr=0.5, 0.5, 0.5
"""]]},
		{"id":"simil_f1_Fshape","cat":"similarity","crit":[["format","=","1"],["origin","=","D"],["numparts","=",5],["numneurons","=",0],["similar_to","<","""//0
p:vr=0.5, 0.5, 0.5
p:1.0, vr=0.5, 0.5, 0.5
p:1.00029632679056, -0.999999956095216, vr=0.5, 0.5, 0.5
p:2.0, vr=0.5, 0.5, 0.5
p:2.00029632679056, -0.999999956095216, vr=0.5, 0.5, 0.5
j:0, 1, dx=1.0, 0.0, 0.0, vr=0.5, 0.5, 0.5
j:1, 2, rz=-1.5705, dx=1.0, 0.0, 0.0, vr=0.5, 0.5, 0.5
j:1, 3, dx=1.0, 0.0, 0.0, vr=0.5, 0.5, 0.5
j:3, 4, rz=-1.5705, dx=1.0, 0.0, 0.0, vr=0.5, 0.5, 0.5
"""]]},
		{"id":"simil_f1_square","cat":"similarity","crit":[["format","=","1"],["origin","=","D"],["numparts","=",5],["numneurons","=",0],["similar_to","<","""//0
p:vr=0.5, 0.5, 0.5
p:1.0, vr=0.5, 0.5, 0.5
p:1.00029632679056, -0.999999956095216, vr=0.5, 0.5, 0.5
p:0.000296502409693367, -1.00059260965032, vr=0.5, 0.5, 0.5
p:-0.000592477857904997, -0.000593004793351115, vr=0.5, 0.5, 0.5
j:0, 1, dx=1.0, 0.0, 0.0, vr=0.5, 0.5, 0.5
j:1, 2, rz=-1.5705, dx=1.0, 0.0, 0.0, vr=0.5, 0.5, 0.5
j:2, 3, rz=-1.5705, dx=1.0, 0.0, 0.0, vr=0.5, 0.5, 0.5
j:3, 4, rz=-1.5705, dx=1.0, 0.0, 0.0, vr=0.5, 0.5, 0.5
"""]]},
		{"id":"simil_f1_hook","cat":"similarity","crit":[["format","=","1"],["origin","=","D"],["numparts","=",7],["numneurons","=",0],["similar_to","<","""//0
p:vr=0.5, 0.5, 0.5
p:1.0, vr=0.5, 0.5, 0.5
p:1.50017107459707, -0.865926611287823, vr=0.5, 0.5, 0.5
p:1.00051328232425, -1.73214949866788, vr=0.5, 0.5, 0.5
p:0.000513457943379536, -1.73274215222298, vr=0.5, 0.5, 0.5
p:-0.500170723298715, -0.867112121173958, vr=0.5, 0.5, 0.5
p:-0.499144334329445, -2.59896503960304, vr=0.5, 0.5, 0.5
j:0, 1, dx=1.0, 0.0, 0.0, vr=0.5, 0.5, 0.5
j:1, 2, rz=-1.047, dx=1.0, 0.0, 0.0, vr=0.5, 0.5, 0.5
j:2, 3, rz=-1.047, dx=1.0, 0.0, 0.0, vr=0.5, 0.5, 0.5
j:3, 4, rz=-1.047, dx=1.0, 0.0, 0.0, vr=0.5, 0.5, 0.5
j:4, 5, rz=-1.047, dx=1.0, 0.0, 0.0, vr=0.5, 0.5, 0.5
j:4, 6, rz=1.047, dx=1.0, 0.0, 0.0, vr=0.5, 0.5, 0.5
"""]]}
	];

	//layout: [ ["cat_id", "Category Title", "category_icon", (added automatically: [ list, of, quest, ids, ... ] ) ]*N ]
	g_quest_categories=[
	 ["bbf9","Bodybuilding f9","cat_buildingf9"],
	 ["bbf1","Bodybuilding f1","cat_buildingf1"],
	 ["run","Running","cat_running"],
	 ["swim","Swimming","cat_swimming"],
	 ["similarity","Look-alike design","cat_similarity"]
	];
  	g_quest_categories_by_id={};
	for(var c in g_quest_categories) { c.add([]); g_quest_categories_by_id[c[0]]=c; }

  	g_quests_by_id={};
	for(var a in g_quests)
            {
	    g_quests_by_id[a->id]=a;
	    // add quest id to its category
	    var c=g_quest_categories_by_id[a->cat];
	    if (c)
	        {
		c[3].add(a->id);
		a->goal=c[3].size;
		}
	    }

}

function getPropertyValue(genotype,propname)
{
	if (propname=="format") //TODO really, format is the only field in Geno not available in Genotype? (...and we wanted these classes to become one since long time ago?)
		return genotype.geno.format;
	else if (propname=="origin")
		{
		if (genotype.data->edited) return "D";
		if (genotype.data->evaluations) return "E";
		return "?"; //other
		}
        else if (propname=="water_velocity")
                {
		var v=genotype.data[propname];
		if (v==null) v=0.0;
		return v;
		}
	else
		return genotype.[propname];
}

function quote(str)
{
	return "\""+str+"\"";
}

function checkQuest(quest,genotype)
{
        if (genotype.data->imported)
                return 0;

	var ok=1;
	for(var crit in quest->crit) // ["name","relation",required_value]
	{
		var actual=null;
		var rel=crit[1];
		var required=crit[2];
		if (crit[0]=="similar_to") //must be handled separately because the result depends on similarity(genotype,required_value), not just on some existing precomputed field of genotype. SimilMeasure.evaluateDistance() could be moved to AL-eval if this calculation becomes costly.
		{
			// SimilMeasureHungarian was not perfect because for some symmetrical objects (e.g. f9 square), the SVD alignment could (and would) rotate the object freely while the variance based on Part coordinates would not change. Then both squares would be compared (including the simil_partgeom component), but one would be rotated instead of perfectly aligned.
			/*
			SimilMeasureHungarian.simil_parts   =1.0;
			SimilMeasureHungarian.simil_partdeg =1.0;
			SimilMeasureHungarian.simil_neuro   =1.0;
			SimilMeasureHungarian.simil_partgeom=1.0;
			SimilMeasureHungarian.simil_fixedZaxis=0;
			const MAX_DISSIMILARITY = 0.1; //these parameter values may need to be adjusted...
			*/
			SimilMeasureDistribution.simil_density = 5;
			SimilMeasureDistribution.simil_bin_num = 64;
			SimilMeasureDistribution.simil_samples_num = 10000; //would be good to increase for even higher accuracy, but then it is (too) slow :|
			const MAX_DISSIMILARITY = 0.02; //these parameter values may need to be adjusted...
			actual=/*SimilMeasureHungarian.*/SimilMeasureDistribution.evaluateDistance(genotype.geno, Geno.newFromString(required));
			required=MAX_DISSIMILARITY;
		} else
		{
			actual=getPropertyValue(genotype,crit[0]);
			if (typeof(actual)=="string")
				actual=quote(actual);
			if (typeof(required)=="string")
				required=quote(required);
			if (rel=="=")
				rel="==";
		}
		var test="return "+actual+rel+required+";";
		//CLI.println(test);
		var testresult=Simulator.eval(test);
		if (typeof(testresult)=="Error")
		{
			Simulator.message("Invalid expression in Quest evaluation: "+test,2);
			Simulator.message(testresult.message,2);
			testresult=0;
		}
		ok=ok && testresult;
	}
	return ok;
}

function isQuestCompleted(id)
{
if (g_quests_state.hasKey(id))
   return g_quests_state[id];
return null;
}

function isQuestCategoryCompleted(cat)
{
for(var q in cat[3])
  if (!isQuestCompleted(q)) return 0;
return 1;
}

function completedQuestCategoriesCount()
{
var count=0;
for(var c in g_quest_categories)
   if (isQuestCategoryCompleted(c)) count++;
return count;
}

function questIDsCompletedByGenotype(g)
{
var ret=[];
for(var q in g_quests)
  {
  var c=isQuestCompleted(q->id);
  if (c!=null)
    if (c->genotype==g)
      ret.add(q->id);
  }
if (ret.size>0)
  return ret;
else
  return null;
}

function categoryIDsFromQuestIDs(ids)
{
if (ids==null) return null;
var cats={};
for(var qid in ids)
  {
  var q=g_quests_by_id[qid];
  if (!cats.hasKey(q->cat)) cats[q->cat]=1;
  else cats[q->cat]++;
  }
return cats;
}

// update g_quests_state, returns list of newly completed quests
// layout: { "quest id"->{id->"string", genotype->"XXX...", genotype_name->"name", time->timestamp }*N }
function updateQuests()
{
var added=[];
if (typeof(g_quests_state)!="Dictionary")
   g_quests_state={};
for(var q in g_quests)
	for(var g in GenePools[0])
    		if (checkQuest(q,g)==1)
		   {
		   if (!g_quests_state.hasKey(q->id))
		      {
		      added.add(q->id);
		      g_quests_state[q->id]={"id":q->id,"genotype":g.genotype,"name":g.name,"data":g.data,"velocity":g.velocity,"distance":g.distance,"vertvel":g.vertvel,"vertpos":g.vertpos};
		      }
		   break;
		   }
Simulator.print("g_quests_state=\n"+g_quests_state);
Simulator.print("added="+added);
saveQuests();
return added;
}

const QUESTS_FILENAME="mobi-quests.data";
const QUESTS_VERSION="1";

function saveQuests()
{
var f=File.createDirect(QUESTS_FILENAME,"persistent quests storage");
f.writeString(QUESTS_VERSION+"\n");
f.writeString(String.serialize(g_quests_state)+"\n");
f.close();
}

function loadQuests()
{
g_quests_state=null;
var fromfile=null;
var f=File.open("scripts_output"+File.pathseparator+QUESTS_FILENAME);
if (f!=null)
   {
   var file_version=f.readLine();
   fromfile=String.deserialize(f.readLine());
   f.close();
   if (file_version==QUESTS_VERSION)
      g_quests_state=fromfile;
   Simulator.print("loaded:\n"+g_quests_state);
   }
if (typeof(g_quests_state)!="Dictionary")
   {
   g_quests_state=null;
   loadHistory();
   if (typeof(fromfile)=="Dictionary") for(var i=0;i<fromfile.size;i++) //temporarily recreate genotypes from old achievement state file - so old achievements won't be lost if genotypes are no longer in "your genotypes"
      {
      var old=fromfile[i];
      var g=Genotype.newFromString(old->genotype);
      if (!g.is_valid) continue;
      g.name=old->name;
      g.data=old->data;
      g.velocity=old->velocity;
      g.distance=old->distance;
      g.vertvel=old->vertvel;
      g.vertpos=old->vertpos;
      g.moveTo(GenePools[0]);
      }
   updateQuests();
   }
}
