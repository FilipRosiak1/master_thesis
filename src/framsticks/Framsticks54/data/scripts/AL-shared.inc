global g_helptxt; //needed in all shows (for the "?" corner button), so let's declare it here once for all shows


function findGenotypeAndName(name, genotype, pool)
{
	for(var g in pool)
		if ((g.name == name) && (g.genotype == genotype))
			return g.index;
	return -1;
}

function setInitialDataDictionary(g)
{
	g.info_timestamp = 0;
	g.data = {"version": 1, "evaluations": 0, "selections": 0, "edited": 0, "imported": 0, "water_velocity": 0.0};
}

function getInitialGenotype()
{
	return "RcqXXX(ddGGbbXX,DDggbbX)";
}

function addInitialGenotype()
{
	var g = GenePools[0].add(getInitialGenotype());
	g.name = "Initial genotype";
	setInitialDataDictionary(g);
}

function loadCurrent()
{
	GenePools[0].clear();
	Simulator.import("scripts_output" + File.pathseparator + "mobi-current.gen");
	if (GenePools[0].size == 0)
		addInitialGenotype();
}

function saveCurrent()
{
	Simulator.export("mobi-current.gen", 2, 0, 0);
}

function saveGenotypeAsCurrent(g)
{
	var f = File.createDirect("mobi-current.gen", "current genotype");
	if (g != null)
		f.writeNameObject("org", g);
	f.close();
}

function loadHistory()
{
	var pool = GenePools[0];
	pool.clear();
	addInitialGenotype();
	Simulator.import("scripts_output" + File.pathseparator + "mobi-history.gen");
	if (pool.size > 1)
		if (pool[0].genotype == pool[1].genotype)
			pool[1].delete();

	for(var i = 0; i < pool.size; i++)
	{
		var g = pool[i];
		if (!g.data.hasKey("version") || (g.data->version != 1))
		{
			Simulator.message("Removing incompatible genotype '"+g.name+"'",1);
			pool[i].delete();
			i--;
		}
	}
}

function saveHistory()
{
	Simulator.export("mobi-history.gen", 2, 0, 0);
}

function addHistory(g)
{
	loadHistory();

	// history limit: since quests are saved independently, we can remove old genotypes to prevent unlimited growth of history
	//if (GenePools[0].size > 49)
	//	GenePools[0][0].delete();

	g.moveTo(GenePools[0]);
	g.info_timestamp = Math.time;
	saveHistory();
}

const SCALE_MAX_SELECTIONS = 100;
function scaleNumberOfEvaluations(evaluations) {return Math.log(evaluations+500)-6.2146;} //nonlinear scale so change is visible immediately for low numbers of evaluations, and slows down for larger numbers
const SCALE_MAX_EVALUATIONS = 5.303313; //scaleEvaluations(100 000 evaluations)
const SCALE_MAX_VELOCITY = 0.01;
const SCALE_MAX_VERTPOS = 1.0;
const SCALE_MAX_WATERVELOCITY = 0.01;

function symbolsForValue(value, sym_max, sym_format, sym_count)
{
	const ASCII_a_MINUS_1 = 96; // 96 is String.code("a") - 1;
	var qv = Math.max(1, Math.min(5 * sym_count - 1, int(float(value) / sym_max * (sym_count * 5) + 0.5)));
	var t = "";
	if (qv >= sym_count) t = (sym_format % String.char(sym_count + ASCII_a_MINUS_1)) * (qv / (sym_count + 1));
	qv = qv % (sym_count + 1);
	if (qv > 0) t += sym_format % String.char(ASCII_a_MINUS_1 + qv);
	return t;
}

const BREADCRUMB_YOUR_CREATURES = "Your creatures";
const BREADCRUMB_CURRENT = "Current";

function makeBreadcrumbPath(nodes)
{
	var txt = "";
	for(var n in nodes)
	{
		if (txt != "") txt += " > ";
		txt += n;
	}
	return txt;
}

function hideTopStatus()
{
	GUI->topstatus.visible = 0;
}

function slideTopStatus()
{
	GUI->topstatus.animSetBegin();
	GUI->topstatus.setRect(0, -GUI.fontHeight(0), GLDisplay.size_x, GUI.fontHeight(0));
	GUI->topstatus.animSetEnd();
	GUI->topstatus.startAnimation(1);
	GUI.setTimeout("topstatus", 1, function hideTopStatus);
}

function displayTopStatus(txt)
{
	if (GUI->topstatus == null)
	{
		var lab = GUI.newLabel("topstatus", txt);
		lab.bgcolor = 0xff000000;
		lab.align = 0;
		lab.font = 0;
	}
	else
		GUI->topstatus.text = txt;
	GUI->topstatus.visible = 1;
	GUI->topstatus.setRect(0, 0, GLDisplay.size_x, GUI.fontHeight(0));
	GUI.setTimeout("topstatus", 3, function slideTopStatus);
}

function ALsharedResetGUI()
{
}

function addHelpButton(place)
{
	var but = GUI.newButton("helpbut", "");
	but.style = 1;
	but.image = "help-icon-shadow";
	but.addTo(place);
}

function handleHelp(elem)
{
	if (String.startsWith(elem.name, "help"))
	{
		if (GUI->help)
		{
			GUI->help.visible = !GUI->help.visible;
		}
		else
		{
			var h = GUI.newBubble("help", g_helptxt);
			h.addTo(GUI->helpbut);
			h.setAutolayout(0, 0, 0, 0, 0, 0, 1, 1);
		}
		return 1;
	}
	return 0;
}

function closePopup()
{
	if (GUI->help && GUI->help.visible)
	{
		handleHelp(GUI->help);
		return 1;
	}
	return 0;
}
