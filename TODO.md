1) Check if the seed is properly set for the model (before init).
2) Set up argument parsing from MyTorch :]
3) Remove redundancies from config (taskname -> datadir)
4) Change mode to bool
5) Append TaskName to accuracy and other dumping things!
6) Fix up logging intelligently.
7) Also keep alpha from args
----

1. Run optimizer.step before schedulers.step
