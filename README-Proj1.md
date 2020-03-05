# ENPM661_Projects
# Project 1 
# Code to solve a 8 puzzle and give the shortest path to reach the goal

Python version -  3.7.6

# Libraries used - numpy, time
Steps to Run:

1) Run the puzzle.py file and enter the puzzle to be solved in the ROW - wise fashion.

	For example, Consider a puzzle like this

	1 2 3
	
	4 5 6 
	
	7 8 0
	
Enter the elements in the row-wise fashion like,

	1
	
	2
	
	3
	
	4
	
	5
	
	6
	
	7	
	
	8	
	
	0

2) The script generates 3 other text files, which gets saved in the same directory as the source file.
   Nodes.txt file contains all the visited nodes.
   nodePath.txt file contains the path taken from the Start node to the Goal node 
   NodesInfo.txt file contains the parent node to every node visited.
   
3) Run puzzle_final.py to visualize the path from the start node to the goal node.
