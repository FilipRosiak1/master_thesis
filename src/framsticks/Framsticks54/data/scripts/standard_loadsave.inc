// traditional experiment load/save handling, does not load/save Creatures

@include "standard_exploadsave.inc"

function onExpLoad()
{
	standardExpLoad(LOADSAVE_EVERYTHING_BUT_CREATURES);
}

function onExpSave()
{
	standardExpSave(LOADSAVE_EVERYTHING_BUT_CREATURES);
}
