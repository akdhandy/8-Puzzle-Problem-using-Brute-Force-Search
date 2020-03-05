# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 03:26:50 2020

@author: Arun
"""

import math
import numpy as np
import cv2 as cv




def boundary_check(i,j):
    if (i<0 or j>199 or j<0 or i>299):
        return 0
    else:
        return 1


def obs_map(x,y):
    circle = ((np.square(x-225))+ (np.square(y-50)) <=np.square(25))
    ellipse = (((np.square(x-150))/np.square(40))+((np.square(y-100))/np.square(20)) -1 <=0)
    if circle or ellipse:
        obj_val = 0
    else:
        obj_val = 1
    
    return obj_val



parent_list=[]

for j in range (300):
    column=[]
    for i in range (200):
        column.append(0)
    parent_list.append(column)
    


x_start=int(input("Please enter start point x coordinate"))
y_start=int(input("Please enter start point y coordinate"))

start_obs=obs_map(x_start,y_start)
start_boundary=boundary_check(x_start,y_start)


while(start_obs and start_boundary!=1):
    print("Incorrect start point! Please enter a valid start point")
    x_start=int(input("Please enter start point x coordinate"))
    y_start=int(input("Please enter start point y coordinate"))
    start_obs=obs_map(x_start,y_start)
    start_boundary=boundary_check(x_start,y_start)


start=[x_start,y_start]


x_goal=int(input("Please enter goal point x coordinate"))
y_goal=int(input("Please enter goal point y coordinate"))

goal_obs=obs_map(x_goal,y_goal)
goal_boundary=boundary_check(x_goal,y_goal)


while(goal_obs and goal_boundary!=1):
    print("Incorrect goal point! Please enter a valid goal point")
    x_goal=int(input("Please enter another goal point x coordinate"))
    y_goal=int(input("Please enter another goal point y coordinate"))
    goal_obs=obs_map(x_goal,y_goal)
    goal_boundary=boundary_check(x_goal,y_goal)
    

goal=[x_goal,y_goal]

resolution=input("Please enter the resolution")


cost_array=np.array(np.ones((300,200)) * np.inf)
visited=np.array(np.zeros((300,200)))

size=(200,300)
mapy=np.ones((size),np.uint8)*255



Q=[]

# append start point and initialize it's cost to zero

Q.append([x_start,y_start])
cost_array[x_start][y_start]=0

# Priority Queue Function

def pop(Q):
    minimum_index=0
    minimum_X = Q[0][0] 
    minimum_Y = Q[0][1]
    for i in range(len(Q)):
        x = Q[i][0]
        y = Q[i][1]
        if cost_array[x,y] < cost_array[minimum_X,minimum_Y]:
            minimum_index = i
            minimum_X = x 
            minimum_Y= y
        
    current_node = Q[minimum_index]
    Q.remove(Q[minimum_index])
    return current_node



# All possible movements
def north(i,j):
    new_node=[i,j+1]     
    return new_node

def south(i,j):
    new_node=[i,j-1]
    return new_node

def east(i,j):
    new_node=[i+1,j]
    return new_node

def west(i,j):
    new_node=[i-1,j]
    return new_node

def NE(i,j):
    new_node=[i+1,j+1]
    return new_node

def SE(i,j):
    new_node=[i+1,j-1]
    return new_node

def NW(i,j):
    new_node=[i-1,j+1]
    return new_node
def SW(i,j):
    new_node=[i-1,j-1]
    return new_node



visited_node=[]
current_node=[x_start,y_start]
while current_node!=goal:
    current_node=pop(Q)
    
    new_north=north(current_node[0],current_node[1])
    status=boundary_check(new_north[0],new_north[1])
    flag=obs_map(new_north[0],new_north[1])
    if (status and flag == 1):
        if visited[new_north[0],new_north[1]]==0:
            visited[new_north[0],new_north[1]]=1
            visited_node.append(new_north)
            Q.append(new_north)
            parent_list[new_north[0]][new_north[1]]=current_node
            cost_array[new_north[0],new_north[1]]=(cost_array[current_node[0],current_node[1]]+1)
        else:
            if cost_array[new_north[0],new_north[1]]>(cost_array[current_node[0],current_node[1]]+1):
                cost_array[new_north[0],new_north[1]]=(cost_array[current_node[0],current_node[1]]+1)
                parent_list[new_north[0]][new_north[1]]=current_node
    
    
    new_south=south(current_node[0],current_node[1])
    status=boundary_check(new_south[0],new_south[1])
    flag=obs_map(new_south[0],new_south[1])
    if (status and flag == 1):
        if visited[new_south[0],new_south[1]]==0:
            visited[new_south[0],new_south[1]]=1
            visited_node.append(new_south)
            Q.append(new_south)
            parent_list[new_south[0]][new_south[1]]=current_node
            cost_array[new_south[0],new_south[1]]=(cost_array[current_node[0],current_node[1]]+1)
        else:
            if cost_array[new_south[0],new_south[1]]>(cost_array[current_node[0],current_node[1]]+1):
                cost_array[new_south[0],new_south[1]]=(cost_array[current_node[0],current_node[1]]+1)
                parent_list[new_south[0]][new_south[1]]=current_node
    
    new_east=east(current_node[0],current_node[1])
    status=boundary_check(new_east[0],new_east[1])
    flag=obs_map(new_east[0],new_east[1])
    if (status and flag == 1):
        if visited[new_east[0],new_east[1]]==0:
            visited[new_east[0],new_east[1]]=1
            visited_node.append(new_east)
            Q.append(new_east)
            #print('pl',parent_list[new_east[0]][new_east[1]])
            parent_list[new_east[0]][new_east[1]]=current_node
            cost_array[new_east[0],new_east[1]]=(cost_array[current_node[0],current_node[1]]+1)
        else:
            if cost_array[new_east[0],new_east[1]]>(cost_array[current_node[0],current_node[1]]+1):
                cost_array[new_east[0],new_east[1]]=(cost_array[current_node[0],current_node[1]]+1)
                parent_list[new_east[0]][new_east[1]]=current_node
    
    
    
    new_west=west(current_node[0],current_node[1])
    status=boundary_check(new_west[0],new_west[1])
    flag=obs_map(new_west[0],new_west[1])
    if (status and flag == 1):
        if visited[new_west[0],new_west[1]]==0:
            visited[new_west[0],new_west[1]]=1
            visited_node.append(new_west)
            Q.append(new_west)
            parent_list[new_west[0]][new_west[1]]=current_node
            cost_array[new_west[0],new_west[1]]=(cost_array[current_node[0],current_node[1]]+1)
        else:
            if cost_array[new_west[0],new_west[1]]>(cost_array[current_node[0],current_node[1]]+1):
                cost_array[new_west[0],new_west[1]]=(cost_array[current_node[0],current_node[1]]+1)
                parent_list[new_west[0]][new_west[1]]=current_node
    
    
    new_NE=NE(current_node[0],current_node[1])
    status=boundary_check(new_NE[0],new_NE[1]) 
    flag=obs_map(new_NE[0],new_NE[1])
    if (status and flag == 1):    
        if visited[new_NE[0],new_NE[1]]==0:
            visited[new_NE[0],new_NE[1]]=1
            visited_node.append(new_NE)
            Q.append(new_NE)
            parent_list[new_NE[0]][new_NE[1]]=current_node
            cost_array[new_NE[0],new_NE[1]]=(cost_array[current_node[0],current_node[1]]+math.sqrt(2))
        else:
            if cost_array[new_NE[0],new_NE[1]]>(cost_array[current_node[0],current_node[1]]+math.sqrt(2)):
                cost_array[new_NE[0],new_NE[1]]=(cost_array[current_node[0],current_node[1]]+math.sqrt(2))
                parent_list[new_NE[0]][new_NE[1]]=current_node
    
    
    new_SE=SE(current_node[0],current_node[1])
    status=boundary_check(new_SE[0],new_SE[1])
    flag=obs_map(new_SE[0],new_SE[1])
    if (status and flag == 1):   
        if visited[new_SE[0],new_SE[1]]==0:
            visited[new_SE[0],new_SE[1]]=1
            visited_node.append(new_SE)
            Q.append(new_SE)
            parent_list[new_SE[0]][new_SE[1]]=current_node
            cost_array[new_SE[0],new_SE[1]]=(cost_array[current_node[0],current_node[1]]+math.sqrt(2))
        else:
            if cost_array[new_SE[0],new_SE[1]]>(cost_array[current_node[0],current_node[1]]+math.sqrt(2)):
                cost_array[new_SE[0],new_SE[1]]=(cost_array[current_node[0],current_node[1]]+math.sqrt(2))
                parent_list[new_SE[0]][new_SE[1]]=current_node
            
    new_NW=NW(current_node[0],current_node[1])
    status=boundary_check(new_NW[0],new_NW[1])
    flag=obs_map(new_NW[0],new_NW[1])
    if (status and flag == 1):
        if visited[new_NW[0],new_NW[1]]==0:
            visited[new_NW[0],new_NW[1]]=1
            visited_node.append(new_NW)
            Q.append(new_NW)
            parent_list[new_NW[0]][new_NW[1]]=current_node
            cost_array[new_NW[0],new_NW[1]]=(cost_array[current_node[0],current_node[1]]+math.sqrt(2))
        else:
            if cost_array[new_NW[0],new_NW[1]]>(cost_array[current_node[0],current_node[1]]+math.sqrt(2)):
                cost_array[new_NW[0],new_NW[1]]=(cost_array[current_node[0],current_node[1]]+math.sqrt(2))
                parent_list[new_NW[0]][new_NW[1]]=current_node
    
    new_SW=SW(current_node[0],current_node[1])
    status=boundary_check(new_SW[0],new_SW[1])
    flag=obs_map(new_SW[0],new_SW[1])
    if (status and flag == 1):   
        if visited[new_SW[0],new_SW[1]]==0:
            visited[new_SW[0],new_SW[1]]=1
            visited_node.append(new_SW)
            Q.append(new_SW)
            parent_list[new_SW[0]][new_SW[1]]=current_node
            cost_array[new_SW[0],new_SW[1]]=(cost_array[current_node[0],current_node[1]]+math.sqrt(2))
        else:
            if cost_array[new_SW[0],new_SW[1]]>(cost_array[current_node[0],current_node[1]]+math.sqrt(2)):
                cost_array[new_SW[0],new_SW[1]]=(cost_array[current_node[0],current_node[1]]+math.sqrt(2))
                parent_list[new_SW[0]][new_SW[1]]=current_node

print("Goal reached")



visited_node


goal=[x_goal,y_goal]
start=[x_start,y_start]
path=[]
def path_find(goal,start):
    GN=goal
    path.append(goal)
    while (GN!=start):
        a=parent_list[GN[0]][GN[1]]
        path.append(a)
        GN=a

path_find(goal,start)
path




print('The cost of the shortest path is',cost_array[x_goal,y_goal])


x = 300
y = 200
image = np.ones((y,x,3),np.uint8)*255

            

            
# Circle
if ((np.square(x-225))+ (np.square(y-50)) <=np.square(25)):
    image[y,x] = (0,0,0)

            
# Ellipse
if (((np.square(x-150))/np.square(40))+((np.square(y-100))/np.square(20)) -1 <=0): 
    image[y,x] = (0,0,0)

            




cv.circle(image,(int(goal[0]),int(goal[1])), (1), (0,0,255), -1);
cv.circle(image,(int(start[0]),int(start[1])), (1), (0,0,255), -1);

for i in visited_node:
    cv.circle(image,(int(i[0]),int(i[1])), (1), (255,0,0));
    pic=cv.resize(image,None,fx=3,fy=3)
    cv.imshow('map',pic)
    cv.waitKey(1)

for i in path:
    cv.circle(image,(int(i[0]),int(i[1])), (1), (153,50,204));
    pic=cv.resize(image,None,fx=3,fy=3)
    cv.imshow('map',pic)
    cv.waitKey(1)
    
    
cv.imwrite('map1.png',image)
cv.waitKey(0) 
cv.destroyAllWindows()