property:
id:evalcount
name:Multiple evaluation
type:d 0 1000 0
help:~
If more than zero:
- each genotype will be evaluated many times
- fitness will be averaged
- fitness standard deviation will be stored in the data["multieval"] field of a genotype (if using standard.expdef)
- there will be no "Unchanged" genotypes ("Unchanged" value is considered zero).~
group:Evaluation

property:
id:evalplan
name:Evaluation plan
type:s 1 -1 :fit,fit_stdev,time
help:~
Genotypes can be evaluated by multiple simulator parameter files (*.sim). Results produced by each parameter file are saved in the selected Genotype.data key as a Dictionary.

Each line in this field defines a single configuration consisting of the sequence of *.sim filenames followed by a colon and a list of associated performance criteria. Multiple *.sim filenames are separated by the plus sign '+' reminding that the settings imports are cumulative (items not specified in the next file keep their previous values). Empty filename means that the evaluation will be performed using current simulator parameters, without importing any file that contains simulator parameters. There are 3 kinds of performance criteria (after ':'):
- genotype fields, listed as plain names: velocity, vertpos, etc.
- data fields, listed as data->field_name (created by script, see foodfinding-usercode.inc)
- special properties:
  - 'fit' and 'fit_stdev' are the fitness average and standard deviation  
  - 'time' is the measured in seconds real time spent on evaluating a given genotype

Example (the first line of this sample plan contains an empty filename):

:velocity,fit
walking.sim:velocity,fit,fit_stdev,time
quicktest.sim:vertpos
roughterrain.sim:velocity,time
walking.sim+reduced_gravity.sim:velocity,data->special_fitness
~
group:Evaluation

property:
id:evalsavedata
name:Save results to key in Genotype.data
type:s 0 -1 evaluation
help:Genotype.data dictionary key where performance results from each of the multiple parameter runs will be saved
group:Evaluation

property:
id:evalsavefile
name:Save results to file name
type:s 0 -1 genos_eval.json
help:File name (in "data/scripts_output" directory) where performance results from each of the multiple parameter runs will be saved
group:Evaluation

property:
id:evalprintprogress
name:Print evaluation progress
type:d 0 1 1
group:Evaluation
