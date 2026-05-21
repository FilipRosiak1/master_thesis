# evolution demo parameters

property:
id:initialgen
name:Initial genotype
type:s 1
help:~
Initial genotype, mutated to create initial population of genotypes.
You need to (re)initialize the experiment to (re)create initial population of genotypes.~

property:
id:delrule
name:Delete genotypes
type:d 0 2 ~Randomly~Inverse-proportionally to fitness~Only the worst

property:
id:MaxCreated
name:Simulated creatures
type:d 0 50
help:You need to (re)initialize the experiment to change this setting and create initial population of given size.

property:
id:rotation
name:Initial orientation
type:d 0 4 ~Always 0 degrees~Randomized 180 degrees~Randomized 90 degrees~Randomized 45 degrees~Random
help:Initial heading of newborn creatures

property:
id:creath
name:Initial elevation
type:f -1 50 0.1
help:~
Vertical position (above the surface) where newborn creatures are placed.
Negative values are only used in the water area:
  0   = at the surface
-0.5 = half depth
-1   = just above the bottom~

property:
id:predefinedSetup
name:Fitness: maximize
type:d 0 2 ~Vertical position of body center~Velocity (including initial movements)~Number of parts in body

property:
id:lifeLengthInSteps
name:Number of steps each creature lives
type:d
help:~
Number of simulation steps for the creature in the center of cratures circle.
When setting this value, consider the performance sampling period used for various fitness criteria.~

property:
id:numberOfIdleSteps
name:Number of idle visualization steps
type:d
help:~
Number of idle visualization steps:
- after parent (parents) approaches (approach) the center and before new creature is born in the world
- after new creature is born in the world and before parent (parents) returns (return) to original place (places)
- after central creature disappearance
- after replacement of circle circumference creature by central creature
If this parameter is set to zero, than simulation speeds up.
If this parameter is set to big value, then some simulation details may be better observed.
It is possible to change this value during simulation to obtain different goals.
~

property:
id:p_nop
name:Unchanged
type:f 0 100 20
group:Selection

property:
id:p_mut
name:Mutated
type:f 0 100 64
group:Selection

property:
id:p_xov
name:Crossed over
type:f 0 100 16
group:Selection

property:
id:xov_mins
name:Minimal similarity
type:f 0 9999 0
help:~
Only genotypes with dissimilarity below this threshold will be crossed over.
Value of 0 means no crossover restrictions.~
group:Selection

property:
id:selrule
name:Selection rule
type:~
d 0 5 3 \~Random\~Fitness-proportional (roulette)\~Tournament (2 genotypes)\~Tournament (3 genotypes)\~Tournament (4 genotypes)\~Tournament (5 genotypes)~
group:Selection

state:
id:totaltestedcr
name:Evaluated creatures
type:d
help:Total number of creatures evaluated in the experiment
flags:16

state:
id:stepType
name:Type of step
type:d
flags:32

state:
id:stepsCounter
name:Resetable steps counter
type:d
help:Number of executed steps of given type
flags:32

state:
id:globals
name:Global variables
type:x
flags:32

state:
id:currentGenotype
name:Index of current genotype in GenePools
type:d
help:Index of current genotype in GenePools
flags:32

state:
id:currentCreature
name:Index of current creature in Populations
type:d
help:Index of current creature in Populations
flags:32

