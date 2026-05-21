global g_active_modal;

function showModal(e)
{
	var gr = GUI.newBlocker("modal_fade");
	gr.bgcolor = 0xaa000000;
	e.addTo(gr);
	g_active_modal = e;
}

function hideModal()
{
	if (g_active_modal != null)
	{
		g_active_modal.addTo(null);
		GUI.removeElement("modal_fade");
		return 1;
	}
	return 0;
}
