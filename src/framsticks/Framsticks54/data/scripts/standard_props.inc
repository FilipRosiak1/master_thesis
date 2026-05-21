# standard evolution parameters

property:
id:initialgen
name:Initial genotype
help:~
The gene pool will be replaced with the supplied genotype when the experiment begins.
Use the empty initial genotype if you want to preserve the current gene pool.~
type:s 1 -1 X

property:
id:capacity
name:Gene pool capacity
type:d 0 100000 100

property:
id:MaxCreated
name:Simulated creatures
type:d 0 100 1

property:
id:placement
name:Initial placement
type:d 0 1 ~Random~Central
help:For 'Central' placement, newborn creatures are placed at the world center, if possible.

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
help:Positive selection: how to choose genotypes for cloning/mutation/crossover
group:Selection

property:
id:delrule
name:Delete genotypes
type:d 0 2 ~Randomly~Inverse-proportionally to fitness~Only the worst
help:~
Negative selection: which genotypes should be removed when we need room for new genotypes in the gene pool.
If multiple-criteria NSGA-II is set as positive selection, this setting is ignored and "worst rank and crowding distance" is used as negative selection.~
group:Selection

property:
id:cr_c
name:Constant
type:f -10000 10000 0
help:Constant value added to total fitness
group:Fitness

property:
id:cr_life
name:Life span
type:f -10000 10000 0
help:Weight of life span in total fitness
group:Fitness

property:
id:cr_v
name:Velocity
type:f -10000 10000 0
help:Weight of horizontal velocity in total fitness
group:Fitness

property:
id:cr_gl
name:Body parts
type:f -10000 10000 0
help:Weight of body size (number of parts) in total fitness
group:Fitness

property:
id:cr_joints
name:Body joints
type:f -10000 10000 0
help:Weight of structure size (number of joints) in total fitness
group:Fitness

property:
id:cr_nnsiz
name:Brain neurons
type:f -10000 10000 0
help:Weight of brain size (number of neurons) in total fitness
group:Fitness

property:
id:cr_nncon
name:Brain connections
type:f -10000 10000 0
help:Weight of brain connections in total fitness
group:Fitness

property:
id:cr_di
name:Distance
type:f -10000 10000 0
help:Weight of distance in total fitness
group:Fitness

property:
id:cr_vpos
name:Vertical position
type:f -10000 10000 0
help:Weight of vertical position in total fitness
group:Fitness

property:
id:cr_vvel
name:Vertical velocity
type:f -10000 10000 0
help:Weight of vertical velocity in total fitness
group:Fitness

property:
id:cr_norm
name:Criteria normalization
type:d 0 1 0
help:Normalize each criterion to 0..1 interval before weighting
group:Fitness

property:
id:cr_simi
name:Similarity speciation
type:d 0 1 0
help:If enabled, fitness of each genotype will be reduced by its phenotypic similarity to all other genotypes in the gene pool
group:Fitness

property:
id:cr_nsga
name:NSGA-II for multiple criteria
type:d 0 1 0
help:~
If enabled, fitness will be replaced with Pareto ranks from NSGA-II (Non-dominated Sorting Genetic Algorithm) method.
Using tournament selection is recommended.
This setting also forces negative selection to be "worst rank and crowding distance" independently from the negative selection setting.~
group:Fitness

state:
id:update_nsga_ranks_needed
name:Update NSGA ranks needed?
type:d 0 1 0
help:~
Only effective when cr_nsga=1. This is a "dirty" flag set each time some operation causes NSGA ranks to expire and require recalculation before being used. Note that Simulator.import() and some user manipulations in GUI cannot be detected by script, so this flag will not be set in such events even though it should.~
flags:32

property:
id:Energy0
name:Starting energy
type:f 0 100000 1000
group:Energy
help:Base starting energy level (for each stick)

property:
id:e_meta
name:Idle metabolism
type:f 0 1 1
group:Energy
help:Each stick consumes this amount of energy in one time step

property:
id:feed
name:Automatic feeding
type:d 0 100 0
group:Energy
help:Number of energy pieces in the world

property:
id:feede0
name:Food's energy
group:Energy
type:f 0 10000 500

property:
id:foodgen
name:Food's genotype
group:Energy
type:s 1
help:~
The default food model is a simple, single part object:
  //0
  m:Vstyle=food
  p:
(this genotype is used when you leave this field blank).
You can specify another genotype to create "intelligent" or mobile food.~

property:
id:feedtrans
name:Ingestion multiplier
group:Energy
type:f 0 100 2

property:
id:aging
name:Aging time
group:Energy
type:d 0 1000000 0
help:Idle metabolism doubles after this period (0 disables aging)
