# use standard_threads.inc to calculate the effective number of threads.

property:
id:threads
type:d
name:Number of threads
group:Parallelized
help:~
Use this setting to set the number of threads:
- positive values (1,2,3,...) are interpreted literally as the number of threads,
- zero (0) means the number of threads equal to the number of CPU cores,
- negative values (-1,-2,-3,...) mean that the number of threads will be the number of CPU cores less 1,2,3,...
~

