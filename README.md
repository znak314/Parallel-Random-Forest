# Parallel Random Forest
This repository implements 2 methods for parallelizing Random Forests in Python:
- forest-level parallelism;
- combined parallelism: forest level + node level

At the forest level, each tree is trained in a separate process. At the node level, several features are processed in parallel to find the best partition.
Other options were also considered, such as parallelization of tree node spanning, parallel calculation of the information gain criterion, but they were less efficient and provided improvements only when building shallow trees.
