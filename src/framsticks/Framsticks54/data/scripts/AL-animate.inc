global g_anim_seq,g_anim_pos,g_anim_obj,g_anim_cb;

function animateSequence(obj, seq, cb)
{
	g_anim_obj = obj;
	g_anim_seq = seq;
	g_anim_cb = cb;
	g_anim_pos = 0;
	animStep();
}

function animStep()
{
	if (g_anim_pos >= (g_anim_seq.size - 1))
	{
		call(g_anim_cb)();
		return;
	}
	animPose(g_anim_obj, g_anim_seq[g_anim_pos]);
	g_anim_obj.animSetBegin();
	animPose(g_anim_obj, g_anim_seq[g_anim_pos + 2]);
	g_anim_obj.animSetEnd();
	g_anim_obj.startAnimation(g_anim_seq[g_anim_pos + 1]);
	GUI.setTimeout("anim_seq", g_anim_seq[g_anim_pos + 1], function animStep);
	g_anim_pos += 2;
}

function animPose(obj, pose)
{
	obj.setRect(pose[0], pose[1], pose[2], pose[3]);
	obj.fgcolor = pose[4];
}
