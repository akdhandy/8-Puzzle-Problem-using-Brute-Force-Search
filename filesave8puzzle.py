# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 18:48:40 2020

@author: Arun
"""

def write_final_path(path_formed):  # Write the final path in the text file
    f = open("Path_file.txt", "a")
    for node in path_formed:
        if node.parent is not None:
            f.write(str(node.node_no) + "\t" + str(node.parent.node_no) + "\t")
    f.close()


def write_nodes_explored(explored):  # Write all the nodes explored by the program
    f = open("Nodes.txt", "a")
    for element in explored:
        f.write('[')
        for i in range(len(element)):
            for j in range(len(element)):
                f.write(str(element[j][i]) + " ")
        f.write(']')
        f.write("\n")
    f.close()


def write_nodes_info(visited):  # Write the information about the nodes explored by the program
    f = open("Node_info.txt", "a")
    for n in visited:
        if n.parent is not None:
            f.write(str(n.node_no) + "\t" + str(n.parent.node_no) + "\t")
    f.close()
