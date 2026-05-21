// Calculate the effective user-selected number of threads (also handles tricks with zero and negative values).
// Include standard_props_threads.inc to create the required ExpProperties.threads field.

function getExpPropertiesThreads()
{
	if (ExpProperties.threads > 0)
		return ExpProperties.threads;
	return Math.max(1, Simulator.cpus + ExpProperties.threads);
}

