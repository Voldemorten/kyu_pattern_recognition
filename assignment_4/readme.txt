To run the EM and the VB algorithm on the dataset x, just run the files "EM.m" and "VB.m" respectively.
This will generate output data in the folder "out/", where to subfolders exists as well. "em/" and "vb/". Each of these subfolders will contain the generated data from each algorithm.

Reading the CSV-files:
All the CSV-files follows the following structure. When a digit is alone it is the iteration number. Thus each csv-file will begin with a single number, 1, to denote the first iteration. Then the actual data will appear. This structure is repeated though all iterations.
In the em folder, The Z_prior is the most important file. The others are the parameters.
Like wise in the vb folder, the Z_gamma, is the most important file. 
