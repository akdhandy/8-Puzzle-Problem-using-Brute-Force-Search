# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 22:48:32 2019

@author: Srujan Panuganti
"""

import numpy as np
import copy
import math
import matplotlib.pyplot as plt
#from dataclasses import dataclass, field
np.set_printoptions(threshold=np.inf)
import argparse
import cv2
from itertools import cycle


# ### To read the input from the user
# def read(start,goal,grid):
#     start_pos = []
#     goal_pos = []
#
#     grid_sz = int(grid)
#     data_start = start.split(",")
#     data_goal = goal.split(",")
#     for element in data_start:
#         start_pos.append(int(element))
#     s_pos = (start_pos,0)
#
#     for element in data_goal:
#         goal_pos.append(int(element))
#     g_pos = [(goal_pos,0)]
#
#     return s_pos, g_pos, grid_sz
#
# #to get the input from the user
# parser = argparse.ArgumentParser()
# parser.add_argument('start_position')
# parser.add_argument('goal_position')
# parser.add_argument('grid_size')
# args = parser.parse_args()
# start_position, goal_position, grid_length  = read(args.start_position,args.goal_position,args.grid_size)
# grid_size = [150,250]


start_position = ([5,0],0)
# goal_position = [([145,240],0)]
goal_position = [([40,30],0)]

grid_size = [150,250]
grid_length = 5

import Queue as queue
# import queue


def convert2nDisplay(imageDiscrete):
    imgCopy = copy.deepcopy(imageDiscrete)
    img = np.rot90(imgCopy,1)
    displayImg = cv2.resize(img,(1000,600),interpolation = cv2.INTER_AREA)
    cv2.imshow("Map", displayImg.astype(np.uint8))
    cv2.waitKey(1)

display_arrary = 255*np.ones((151,251,3))
#use this to update the imageDiscrete using the ID's given below

# set up color map for display
# 0 - empty, white
# 1 - startPt, red
# 2 - goalPt, green
# 3 - visited, yellow
# 4 - obstacle, black
# 5 - path, blue

def onlyUpdateDisplay(imageDiscrete,pt,val):
    red =0
    green =0
    blue =0
    if val ==0:
        red,green,blue = 255,255,255
    elif val == 1:
        red,green,blue = 255,0,0
    elif val == 2:
        red,green,blue = 0,255,0
    elif val == 3:
        red,green,blue = 255,255,0
    elif val == 4:
        red,green,blue = 0,0,0
    elif val == 5:
        red,green,blue = 0,0,255
    else:
        # to catch errors
        red,green,blue = 255,0,255
    imageDiscrete[pt[0],pt[1],0] = blue
    imageDiscrete[pt[0],pt[1],1] = green
    imageDiscrete[pt[0],pt[1],2] = red
    return imageDiscrete

# use this to update and display the imageDiscrete
def updateAndDisplay(imageDiscrete,pt,val):
    red =0
    green =0
    blue =0
    if val ==0:
        red,green,blue = 255,255,255
    elif val == 1:
        red,green,blue = 255,0,0
    elif val == 2:
        red,green,blue = 0,255,0
    elif val == 3:
        red,green,blue = 255,255,0
    elif val == 4:
        red,green,blue = 0,0,0
    elif val == 5:
        red,green,blue = 0,0,255
    else:
        # to catch errors
        red,green,blue = 255,0,255
    imageDiscrete[pt[0],pt[1],0] = blue
    imageDiscrete[pt[0],pt[1],1] = green
    imageDiscrete[pt[0],pt[1],2] = red
    convert2nDisplay(imageDiscrete)
    return imageDiscrete


#### solving linear equations
def solve(lin_eqn1,lin_eqn2):
	# lin_eqn1 = [a1,b1,c1], lin_eqn2 = [a2,b2,c2]

	a1 = lin_eqn1[0]
	b1 = lin_eqn1[1]
	c1 = lin_eqn1[2]

	a2 = lin_eqn2[0]
	b2 = lin_eqn2[1]
	c2 = lin_eqn2[2]

	A = np.array([[a1,b1], [a2,b2]])
	B = np.array([-c1,-c2])

	vertex_t = np.linalg.solve(A,B)
	vertex = list([(vertex_t[0],vertex_t[1])])

	return vertex

#### polygon
def polygon(rect):

	pts = np.array(get_vertices(rect), np.int32)
	corrected_coordinates = correct_y_axis(pts)
	return corrected_coordinates

### circle
def circle(img,cir_center,cir_radius):
	#cv2.circle(img,(190,150 - 130),20,1,-1)
	#img = np.copy(map)
	print(cir_center)
	cv2.circle(img,cir_center,cir_radius,1,-1)

### ellipse
def ellipse(img,eli_center,axes):
	cv2.ellipse(img,eli_center,axes,0,0,360,1,-1)

### obtaining vertices given half planes
def get_vertices(rect1):

	rect = cycle(rect1)
	temp = []
	verts = []

	for i in range (0,len(rect1)+1):
		## here we are capturing each element in cycled rect using the next command in itertools
		temp.append(next(rect))

	for i in range (0,len(temp)-1):
		verts.append(solve(temp[i],temp[i+1]))

	return verts

## Y-Axis correction
def correct_y_axis(coord):
	for i in range (0, len(coord)):
		#coord[i][:,1]=150-(coord[i][:,1])
		coord[i][:,1]=np.shape(map)[0]-(coord[i][:,1])

	return coord

## to generate map with circular robot
def circular_robot_map(map_passed, ellipse_center, circle_bot_ellipse_axes, circle_center, circle_obstacle_radius):

	#eli_center = tuple(list(correct_y_axis(ellipse_center)[0][0]))
	map = np.copy(map_passed)
	##### Half-planes defining the rectangle

	equation_1 = [0,1,-117.5]   ### a1*x+b1*y+c1 = [a1,b1,c1]
	equation_2 = [1,0,-45]
	equation_3 = [0,1,-62.5]
	equation_4 = [1,0,-105]

	rect1 = [equation_1,equation_2,equation_3,equation_4]

	rect1_vertices = polygon(rect1)
	#cv2.polylines(map,[rect1_vertices],True,(255,255,255))
	cv2.fillPoly(map,[rect1_vertices],True,1)


	## half-plane equations of polygon
	## part 1 triangle ABF

	p1_eqn1 = [2/19,1,-1409/19]  	#AB
	p1_eqn2 = [-47/15,1,6806/15]	#BF
	p1_eqn3 = [41/25,1,-1264/5]		#FA

	part1 = [p1_eqn1,p1_eqn2,p1_eqn3]

	## part 2 triangle BCD
	p2_eqn1 = [-38/7,1,5795/7]		#BC
	p2_eqn2 = [43/28,1,-4985/14]	#CD
	p2_eqn3 = [1/7,1,-562/7]		#DB

	part2 = [p2_eqn1,p2_eqn2,p2_eqn3]

	## part 3 polygon BDEF
	p3_eqn1 = [1/7,1,-562/7]		#BD
	p3_eqn2 = [-37/20, 1, 3143/10]	#DE
	p3_eqn3 = [0, 1, -10]			#EF
	p3_eqn4 = [-47/15, 1, 6806/15]	#FB

	part3 = [p3_eqn1,p3_eqn2,p3_eqn3,p3_eqn4]

	part1_vertices = polygon(part1)
	#print(part1_vertices)
	cv2.fillPoly(map,[part1_vertices],True,1)

	part2_vertices = polygon(part2)
	#print(part2_vertices)
	cv2.fillPoly(map,[part2_vertices],True,1)

	part3_vertices = polygon(part3)
	#print(part3_vertices)
	cv2.fillPoly(map,[part3_vertices],True,1)

	### Other way to plot the polygon###
	# ABCDEF
	polygon_sys_eqns = [p1_eqn1,p2_eqn1,p2_eqn2,p3_eqn2,p3_eqn3,p1_eqn3]
	polygon_circular_vertices = polygon(polygon_sys_eqns)
	#cv2.polylines(map,[polygon_circular_vertices],True,(255,255,255))
	#cv2.fillPoly(map,[polygon_circular_vertices],True,255)


	### plotting the polygon directly using the vertices
	polygon_vertices_circular = np.array([[[116.39,61.90],[163,57],[170,95],[198,52],[175.29,10],[148.04,10]]])
	polygon_vertices_circular_corrected = correct_y_axis(polygon_vertices_circular)
	cv2.polylines(map,np.int32([polygon_vertices_circular_corrected]),True,(255,255,255))

	#### Plotting the ellipse
	ellipse(map,ellipse_center,circle_bot_ellipse_axes)

	## Plotting the circle obstacle
	circle(map, circle_center, circle_obstacle_radius)

	return map

## to generate map with point robot
def point_robot_map(map_passed,ellipse_center,point_bot_ellipse_axes,circle_center,point_bot_radius):

	map_point_robot =	np.copy(map_passed)

	#eli_center = tuple(list(correct_y_axis(ellipse_center)[0][0]))

	rectangle_point_vertices = np.array([[50,150 - 112.5],[100,150 -112.5],[100,150 - 67.5],[50,150 - 67.5]])
	#cv2.polylines(map_point_robot,np.int32([rectangle_point_vertices]),True,(255,255,255))
	cv2.fillPoly(map_point_robot,np.int32([rectangle_point_vertices]),True,1)

	polygon_point_vertices = np.array([[125,150 - 56],[163,150 -52],[170,150 - 90],[193,150 - 52],[173,150 - 15],[150,150 - 15]])
	#cv2.polylines(map_point_robot,[polygon_point_vertices],True,(255,255,255))
	cv2.fillPoly(map_point_robot,[polygon_point_vertices],True,1)

	#### Plotting the ellipse
	ellipse(map_point_robot,ellipse_center,point_bot_ellipse_axes)

	## Plotting the circle obstacle
	circle(map_point_robot,circle_center, point_bot_radius)

	#cv2.imshow('map with obstacles for point robot', map_point_robot)
	return map_point_robot

### map grid size
map = np.zeros([150,250])
#print(np.shape(map)[0])
#print('map image= ', map)

####### ellipse obstacle parameters
ellipse_center = np.array([[[140,120]]])
ellipse_center_corrected = tuple(list(correct_y_axis(ellipse_center)[0][0]))
#print(ellipse_center_corrected)
ellipse_obstacle_axes = (30, 12)
ellipse_obstacle_axes_lengths = ((np.int(ellipse_obstacle_axes[0]/2), np.int(ellipse_obstacle_axes[1]/2)))
#print(ellipse_obstacle_axes_lengths)

#### circle obstacle parameter
circle_center = np.array([[[190,130]]])
circle_center_corrected = tuple(list(correct_y_axis(circle_center)[0][0]))
#print(circle_center_corrected)
circle_obstacle_diameter = 30
circle_obstacle_radius = np.int(circle_obstacle_diameter/2)

#### robot specification
circle_robot_diameter = 10

### enlarged ellipse parameters
enlarged_ellipse_obstacle_axes = tuple(np.int32(((ellipse_obstacle_axes[0] / 2 + circle_robot_diameter / 2), (ellipse_obstacle_axes[1] / 2 + circle_robot_diameter / 2))))

#### enlarged circle parameters
enlarged_circle_obstacle_radius = np.int32((circle_robot_diameter + circle_obstacle_diameter)/2)

##########  For circular robot  ##################
updated_circular_map = circular_robot_map(map, ellipse_center_corrected, enlarged_ellipse_obstacle_axes, circle_center_corrected, enlarged_circle_obstacle_radius)

#### for point robot  ########
updated_map_point_robot = point_robot_map(map, ellipse_center_corrected,ellipse_obstacle_axes_lengths,circle_center_corrected,circle_obstacle_radius)

#cv2.imshow('obs map',updated_map_point_robot)

### To generated up node
def up(position):
    current_position = copy.copy(position)
    new_node = []

    row = current_position[0][0][0]
    col = current_position[0][0][1]

    if row == 0:
        #return 0
        #new_node.append(([row, col], 1 ))
        status = False
    else:
        #x = [row-1, col], 1 + current_position[1]
        #new_node.append(x)
        new_node.append(([row-1, col], 1 ))  ##+ current_position[0][1]
        #print('new_node',new_node)
        status = True

    #return current_position.loc, current_position.cost

    return new_node , status

### To generated down node
def down(position):
    current_position = copy.copy(position)

    new_node = []

    row = current_position[0][0][0]
    col = current_position[0][0][1]
    if row >= grid_size[0]:
        #return 0
        status = False
    else:
        new_node.append(([row+1, col], 1 )) ##+ current_position[0][1]
        status = True

    #return current_position.loc, current_position.cost
    return new_node, status

### To generated left node
def left(position):
    current_position = copy.copy(position)

    new_node = []

    row = current_position[0][0][0]
    col = current_position[0][0][1]
    if col == 0:
        #return 0

        status = False
    else:
        new_node.append(([row, col-1], 1 )) ##+ current_position[0][1]
        #print('new_node-left',new_node)
        status = True

    #return current_position.loc, current_position.cost
    return new_node, status

### To generated right node
def right(position):
    current_position = copy.copy(position)

    new_node = []
    row = current_position[0][0][0]
    col = current_position[0][0][1]
    if col >= grid_size[1]:
        #return 0
        status = False
    else:
        new_node.append(([row, col+1], 1 ))  #+ current_position[0][1]
        status = True
    #return current_position.loc, current_position.cost
    return new_node, status

### To generated up_left node
def up_left(position):
    current_position = copy.copy(position)
    new_node = []
    row = current_position[0][0][0]
    col = current_position[0][0][1]
    if row == 0 or col == 0:
        #return 0
        status = False
    else:
        new_node.append(([row-1, col-1], np.sqrt(2))) ##+ current_position[0][1]
        status = True
    #return current_position.loc, current_position.cost
    return new_node, status

### To generated right node
def up_right(position):
    current_position = copy.copy(position)
    new_node = []
    row = current_position[0][0][0]
    col = current_position[0][0][1]

    if row == 0 or col >= grid_size[1]:
        #return 0
        status = False
    else:
        new_node.append(([row-1, col+1],np.sqrt(2)))  ##+ current_position[0][1]
        #current_position[1] =
        status = True
    #return current_position.loc, current_position.cost
    return new_node, status

### To generated down_left node
def down_left(position):
    current_position = copy.copy(position)
    new_node = []
    row = current_position[0][0][0]
    col = current_position[0][0][1]
    if row >= grid_size[0] or col == 0:
        #return 0
        status = False
    else:
        new_node.append(([row+1, col-1], np.sqrt(2)))  ###+ current_position[0][1]
        status = True

    #return current_position.loc, current_position.cost
    return new_node, status

### To generated down_right node
def down_right(position):
    current_position = copy.copy(position)

    new_node = []
    row = current_position[0][0][0]
    col = current_position[0][0][1]
    if row >= grid_size[0] or col >= grid_size[1]:
       #return 0
       status = False
    else:
        new_node.append(([row+1, col+1], np.sqrt(2)))  #+ current_position[0][1]
        status = True

    return new_node, status
    #return current_position.loc, current_position.cost #, status

## To check if the node is in the obstacle space
def if_obstacle(node,obstacle_space_set):
    return str(node[0][0]) in obstacle_space_set

## to check if the node is visited
def is_visited_check(node, node_check_set):
    return str(node[0][0]) in node_check_set

## To generate all the next nodes from a current node
def explored(node):

    current_node = copy.copy(node)

    new_nodes = []


    up_loc, status  = up(current_node)
    #print(up_loc)
    if status == True:
        new_nodes.append(up_loc)

    up_right_loc,status  = (up_right(current_node))
    #print(up_right_loc)
    if status == True:
        new_nodes.append(up_right_loc)


    right_loc, status  = (right(current_node))
    #print('right',right_loc)
    if status == True:
        new_nodes.append(right_loc)

    down_right_loc, status  = (down_right(current_node))
    #print(down_right_loc)
    if status == True:
        new_nodes.append(down_right_loc)


    down_loc,  status  = (down(current_node))
    #print(down_loc)
    if status == True:
        new_nodes.append(down_loc)

    down_left_loc, status  = (down_left(current_node))
    #print(down_left_loc)
    if status == True:
        new_nodes.append(down_left_loc)


    left_loc, status  = (left(current_node))
    #print('left_loc',left_loc)
    if status == True:
        new_nodes.append(left_loc)



    up_left_loc,status  = (up_left(current_node))
    #print(up_left_loc)
    if status == True:
        new_nodes.append(up_left_loc)



    return new_nodes

## To generate the obstacle_set from the map
def obstacle_list(obstacle_space):
    loc = []
    #print(obstacle_space)
    obstacle_space_set = set([])
    A = np.zeros([150,250])
    for i in range(0,A.shape[0]):
        for j in range(0,A.shape[1]):
            #loc.append([i,j])
            if obstacle_space[i,j] == 1:
                obstacle_space_set.add(str([i,j]))

    return obstacle_space_set

## To generate all the possible nodes of the map
def node_info_list():
    loc = []
    A = np.zeros([150,250])
    for i in range(0,A.shape[0]):
        for j in range(0,A.shape[1]):
            loc.append([i,j])
    distance = {}
    for node in loc:
        distance[str(node)] = 9999999 #math.inf
    #print(distance[str([5,5])])
    return distance

## To check if the entered start and goal position are in the obstacle space or not
def valid_start_goal(start,goal,obstacle_set):
    status = True
    if if_obstacle([start],obstacle_set) or if_obstacle(goal,obstacle_set):
        status = False
    return status


point_robot_obstacle_space_set = obstacle_list(updated_map_point_robot)
circular_robot_obstacle_space_set = obstacle_list(updated_circular_map)


map_new = np.zeros([grid_size[0] + 1, grid_size[1] + 1])

# nodes_list = []
# nodes_list.append([start_position])

## To capture the parent nodes from the visited node
node_info = []
node_info.append([None, start_position[0], 0])

# q = queue.Queue(maxsize=0)
# #q = queue.PriorityQueue(maxsize=0)
# q.put(nodes_list[0])

q = queue.PriorityQueue()
q.put([start_position[1],start_position[0]])


# node_check_set = set([])            #visited nodes
# node_check_set.add(str(nodes_list[0][0][0]))
#
# node_info_dict = node_info_list()
# node_info_dict[str(start_position[0])] = 0

node_check_set = set([])            #visited nodes
node_check_set.add(str(start_position[0]))

node_info_dict = node_info_list()
node_info_dict[str(start_position[0])] = 0


node_info_parent_dict = {}

iter1 = iter2 = 0

is_a_vaid_input = valid_start_goal(start_position,goal_position,point_robot_obstacle_space_set)

goal_reached = False

visited_node = []
def display_map(start_position,goal_position,node_path,visited_node,updated_map_point_robot):
    display_map = updated_map_point_robot
    print(display_map.shape)

    plt.figure(figsize=(150,250))

    for i in range(0,display_map.shape[0]):
        for j in range(0,display_map.shape[1]):
            if display_map[i,j] == 1:
                plt.plot(j,150-i,'k.')

    for node in node_path:
        plt.plot(start_position[0][1],150-start_position[0][0],'bo')
        plt.plot(goal_position[0][0][1],150-goal_position[0][0][0],'go')
        plt.plot(node[1],150-node[0],'r+')

    for node in visited_node:
        plt.plot(node[1],150-node[0],'b+',label = 'explored nodes')

    plt.xlim(0,250)
    plt.ylim(0,150)
    plt.grid(True)
    plt.show()



while not q.empty() and is_a_vaid_input == True:# and :

    # node = q.get()
    node = q.get()
    print('node in while',node)
    node = [(node[1],node[0])]


    iter1+=1
    if node[0][0] == goal_position[0][0]:
        print('goal reached')
        goal_reached = True
        break
    explored_nodes = explored(node)
    for action in explored_nodes:
        iter2 += 1
        if is_visited_check(action,node_check_set) == False:
            if if_obstacle(action,point_robot_obstacle_space_set) == False:

                node_check_set.add(str(action[0][0])) ## marked as visited --> added to visited nodes
                cost = action[0][1] + node_info_dict[str(node[0][0])]
                visited_node.append(action[0][0])
                node_info_dict[str(action[0][0])] = cost
                # q.put([(action[0][0],cost)])
                q.put([cost,action[0][0]])

                node_info.append([node[0][0],action[0][0],cost])        #--> parent is updated to the node info
                node_info_parent_dict[str(action[0][0])] = node[0][0]
                # display_arrary = updateAndDisplay(display_arrary,action[0][0],3)

        else:
            if if_obstacle(action,point_robot_obstacle_space_set) == False:
                temp = action[0][1] + node_info_dict[str(node[0][0])]
                if node_info_dict[str(action[0][0])] > temp:
                    node_info_dict[str(action[0][0])] = temp
                    node_info.append([node[0][0],action[0][0],temp])        #--> parent is updated to the node info
                    node_info_parent_dict[str(action[0][0])] = node[0][0]

if is_a_vaid_input == True:
    if goal_reached:
        node_path = []
        node_path.append(goal_position[0][0])

        parent = node_info_parent_dict[str(goal_position[0][0])]

        while parent != start_position[0]:
            parent =  node_info_parent_dict[str(parent)]
            node_path.append(parent)
            # display_arrary = updateAndDisplay(display_arrary,parent,5)


        print(node_path)
        display_map(start_position,goal_position,node_path,visited_node,updated_map_point_robot)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


    else:
        print('cannot reach the goal')
else:
    print('enter valid inputs: your goal and start positions might be in obstacle space')
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
