global g_hide_screenshot;

function handleSharing(elem)
{
	if (elem.name == "share")
	{
		var payload_and_title = getSharingTitleAndPayload();
		Sharing.select(payload_and_title[0], function sharing_function, payload_and_title[1]);
		return 1;
	}
	return 0;
}

function sharing_function()
{
	if (Sharing.payload & 2)
		{
		Simulator.export("share.gen", 2, 0, 0);
		Sharing.attach_file = "scripts_output" + File.pathseparator + "share.gen";
		}

	if (Sharing.payload & 1)
		{
	var edgebuttons = ["top_left", "top_center", "top_right", "bottom_left", "bottom_center", "bottom_right"];
	for(var name in g_hide_screenshot) GUI[name].visible = 0;
	for(var name in edgebuttons) GUI.[name].visible = 0;
	var thinspace = String.char(226) + String.char(128) + String.char(137); //utf-8
	var lab = GUI.newLabel("", "www.framsticks.com" + thinspace); //thinspace creates more or less the same right margin as the bottom "margin" (due to descender height)
	lab.setAutolayout(0, 0, 0, 0, 0, 0, 1, 1);
	lab.align = 1;
	lab.valign = 1;
	lab.fgcolor = 0x66000000;
	Sharing.addScreenshot();
	for(var name in g_hide_screenshot) GUI[name].visible = 1;
	for(var name in edgebuttons) GUI.[name].visible = 1;
	GUI.removeElement(lab);
		}

	Simulator.print("Sharing (service/message/url/payload):");
	Simulator.print(Sharing.service);
	Simulator.print(Sharing.message);
	Simulator.print(Sharing.url);
	Simulator.print(Sharing.payload);
}
