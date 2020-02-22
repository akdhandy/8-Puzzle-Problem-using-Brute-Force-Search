# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 22:02:02 2020

@author: Arun
"""

import time
import numpy as np
import copy

initial_state = []
temp_index = []
temp_index.append(0)
count_initial = 1


def get_initial():
    print("Enter number from 0-8")
    initial_state = np.zeros(9)
    for i in range(9):
        states = int(input("Enter the number: "))
        initial_state[i] = np.array(states)
    print(initial_state)
    initial_state = np.reshape(initial_state,(3,3))
    return initial_state


def find_index(puzzle):
    for i in range(3):
        for j in range(3):
            if puzzle[i][j] == 0:
                return i,j


def move_left(data):
    temp_arr = copy.deepcopy(data)
    i, j = find_index(temp_arr)
    if j != 0:
        temp = temp_arr[i, j - 1]
        temp_arr[i, j] = temp
        temp_arr[i, j - 1] = 0
        print("Left motion")
        return temp_arr
    else:
        print("No left motion")
        return temp_arr


def move_right(data):
    temp_arr = copy.deepcopy(data)
    i, j = find_index(data)
    if j != 2:
        temp = temp_arr[i, j + 1]
        temp_arr[i, j] = temp
        temp_arr[i, j + 1] = 0
        print("Right motion")
        return temp_arr
    else:
        print("No right motion")
        return temp_arr

def move_up(data):
    temp_arr = copy.deepcopy(data)
    i, j = find_index(data)
    if i != 0:
        temp_arr = np.copy(data)
        temp = temp_arr[i - 1, j]
        temp_arr[i, j] = temp
        temp_arr[i - 1, j] = 0
        print("Up motion")
        return temp_arr
    else:
        print("No up motion")
        return temp_arr


def move_down(data):
    temp_arr = copy.deepcopy(data)
    i, j = find_index(data)
    if i != 2:
        temp_arr = np.copy(data)
        temp = temp_arr[i + 1, j]
        temp_arr[i, j] = temp
        temp_arr[i + 1, j] = 0
        print("Down motion")
        return temp_arr
       
    else:
        print("No down motion")
        return temp_arr

#Performing solvability test
def sol(ip_node):
    z = []
    inv = 0
    for i in range(3):
        for j in range(3):
            if (ip_node[i][j]!=0):
                z.append(ip_node[i][j])            
    for i in range(7):
        j=i+1
        while(j<8):
            if z[i]>z[j]:
                inv += 1
            j += 1
    if inv % 2:
        print("Unsolvable")
        return 0
    else:
        print("Solvable")
        return 1
   

 #to check and append in the parent node
def check_and_append(p_node, new_node, count):
    stat=False
    global count_initial
    q = 1
   
    for l in p_node:
        if (l == new_node).all():  
            q = 0
            print("Not equal, Not appended")
           
   
    if q == 1:
        print("Appended")
        visited_nodes.append(new_node)
        temp_index.append(count_initial)
        stat=True
    return count, stat


# to check if the goal is reached
def goal_check(B,goal_node):
    status = np.array_equal(B,goal_node)
    return status

# initializing the goal node
goal_node = np.array([[1,2,3],[4,5,6],[7,8,0]])

start_time=time.time()


visited_nodes=[]
initial_array = get_initial()
visited_nodes.append(initial_array)
child_node_number=1
print("child index total")
print(visited_nodes)
print("child index 1.")
print(visited_nodes[0])
i=0



#bfs takes place
if (sol(initial_array)):

    while (i < child_node_number):

        node_list = visited_nodes[i]
       

        new_node = move_left(node_list)
        print(new_node)
        child_node_number, status = check_and_append(visited_nodes, new_node, child_node_number)
        print(status)
        if status == True:
            child_node_number += 1



        if goal_check(new_node,goal_node):
            found = True
            break




        new_node = move_up(node_list)
        print(new_node)
        child_node_number, status=check_and_append(visited_nodes, new_node, child_node_number)
        print(status)
        if status == True:
            child_node_number += 1
           


        if goal_check(new_node,goal_node):
            found = True      
            break



        new_node = move_right(node_list)
        print(new_node)
        child_node_number, status=check_and_append(visited_nodes, new_node, child_node_number)
        print(status)
        if status == True:
            child_node_number += 1
           

        if goal_check(new_node,goal_node):
            found = True
            break



        new_node = move_down(node_list)
        print(new_node)
        child_node_number, status=check_and_append(visited_nodes, new_node, child_node_number)
        print(status)
        if status == True:
            child_node_number += 1

        if goal_check(new_node,goal_node):
            found = True
            goal_key = i
            break


            print(visited_nodes)    
            print(child_node_number)
           
           
        i+=1
        count_initial += 1
     
    final_path = []
    start_index = 0
    goal_index=len(visited_nodes)-1
    path = []
    path.append(goal_index + 1)
    
    
    while(goal_index != start_index):
        path.append(temp_index[goal_index])
        goal_index = (temp_index[goal_index]-1)
        
    path.sort()  
    
if (sol(initial_array)):
    final_path = []
    
    for i in path:
        final_path.append(i-1)    
    
    shortest_path=[]
    for elements in final_path:
        shortest_path.append(visited_nodes[elements]) 
    
    
    def write_nodes_explored(visited_nodes):  # Write all the nodes explored by the program
        f = open("Nodes.txt","w+")
        for element in visited_nodes:
            for i in range(len(element)):
                for j in range(len(element)):
                    f.write(str(element[j][i]) + " ")
            f.write("\n")   
        f.close()
        print("Text file created")
        
    
    def write_shortest_path(shortest_path):  # Write the shortest path by the program
        f = open("Nodepath.txt","w+")
        for element in shortest_path:
            for i in range(len(element)):
                for j in range(len(element)):
                    f.write(str(element[j][i]) + " ")
            f.write("\n")
        f.close() 
        print("Text file created")
       
    
    
    def write_parent_nodes(temp_index):#Writing all elements and their parent nodes
        w = 1
        f = open("nodesInfo.txt","w+")
        for element in temp_index:
            f.write(str(w) + " ")
            f.write(str(element))
            f.write("\n")
            w += 1
        f.close()
        print("Text file created")
       
    write_nodes_explored(visited_nodes)
    write_shortest_path(shortest_path)
    write_parent_nodes(temp_index)
    print("total time:")
    print(time.time()-start_time)