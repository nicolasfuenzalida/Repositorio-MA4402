#!/usr/bin/env python
# coding: utf-8

# In[184]:


get_ipython().system('pip install networkx==2.8.8')

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy
import time


# In[7]:


from collections	import defaultdict, deque
from copy 			import deepcopy
from os				import system
from random         import randrange, seed


# In[185]:


def convert(a):
    n = len(a)
    adjList = []
    for k in range(n):
        adjList.append([])
    for i in range(n):
        for j in range(len(a[i])):
                       if a[i][j]== 1:
                           adjList[i].append(j)
    return adjList

def timeSince(since):
    now = time.time_ns()
    s = now - since
    return s*10**(-9)


# In[9]:


def adjacenecy_matrix_to_adjacency_list(matrix):
	adjacency_list = defaultdict(list)
	for row in range(0, len(matrix)):
		for element in range(0, len(matrix[row])):
			if matrix[row][element] == 1:
				adjacency_list[row].append(element)
			# if row not in adjacency_list:
			# 	adjacency_list[row].append(None)

	return dict(adjacency_list)


# In[10]:


# PS clear output
def console_clear():
	system('cls')

# color first vertex, for the rest check connections and assign first available color
def greedy_coloring(matrix):
	colors = list(range(len(matrix)))
	result = {0: colors[0]}
	for vertex in range(1, len(matrix)):
		temp = colors.copy()
		for edge in range(0, len(matrix[vertex])):
			if matrix[vertex][edge] == 1 and edge in result and result[edge] in temp:
				temp.remove(result[edge])
		result[vertex] = temp[0]
	return result, len(set(result.values()))


# In[182]:


def tabu_coloring(adjacency_list, number_of_colors, previous_solution, is_first_solution, max_iterations, TIME, tabu_size = 7, reps = 40):
    start=time.time_ns()
    colors = list(range(number_of_colors))
    iterations = 0
    tabu_list = deque()
    aspiration_dict = dict()
    ITE = 0
    

    solution = deepcopy(previous_solution)
    if not is_first_solution:
        for i in range(len(adjacency_list)):
            if solution[i] >= number_of_colors:
                solution[i] = colors[randrange(0, len(colors))]

    while iterations < max_iterations:
        hora = timeSince(start)
        if hora > TIME:
            if conflict_count != 0:
                print("No coloring found with {} colors.".format(number_of_colors))
                return False, previous_solution,ITE,-1
            else:
                print("Found coloring:", len(set(solution.values())))
                return True, solution,ITE,len(set(solution.values()))
            
        candidates = set()
        conflict_count = 0

        for vertice, edges in adjacency_list.items():
            for edge in edges:
                if solution[vertice] == solution[edge]:
                    candidates.add(vertice)
                    candidates.add(edge)
                    conflict_count += 1

        candidates = list(candidates)

        if conflict_count == 0:
            ITE = iterations
            # Found a valid coloring.
            break

        new_solution = None
        for _ in range(reps):
            vertice = candidates[randrange(0, len(candidates))]
            new_color = colors[randrange(0, len(colors))]
            if solution[vertice] == new_color:
                new_color = colors[-1]

            new_solution = deepcopy(solution)
            new_solution[vertice] = new_color
            new_conflicts = 0
        
            for vertice, edges in adjacency_list.items():
                for edge in edges:
                    if vertice is not None and edge is not None and new_solution[vertice] == new_solution[edge]:
                        new_conflicts += 1
            
            if new_conflicts < conflict_count:
                if new_conflicts <= aspiration_dict.setdefault(conflict_count, conflict_count - 1):
                    aspiration_dict[conflict_count] = new_conflicts - 1
                    if (vertice, new_color) in tabu_list:
                        tabu_list.remove((vertice, new_color))
                        break
                else:
                    if (vertice, new_color) in tabu_list:
                        continue
                break

        tabu_list.append((vertice, solution[vertice]))
        if len(tabu_list) > tabu_size:
            tabu_list.popleft()

        solution = deepcopy(new_solution)
        iterations += 1
        ITE = iterations
        print(iterations)

    if conflict_count != 0:
        print("No coloring found with {} colors.".format(number_of_colors))
        return False, previous_solution,ITE,-1
    else:
        print("Found coloring:", len(set(solution.values())))
        return True, solution,ITE,len(set(solution.values()))


# In[2]:


def tabu_search(adjacency_list, greedy_result_dict, greedy_result_number, max_iterations,TIME):
    numero_final = 0
    start = time.time_ns()
    colors_used = []
    first_coloring = True
    result = greedy_result_dict
    for v in range(greedy_result_number, 1, -1):
        hora = timeSince(start)
        status, result, ite_number,use_less= tabu_coloring(adjacency_list, v, result, first_coloring, max_iterations,TIME-hora)
        if not status:
            print("entre aqui y debo terminar")
            j=0
            while j < max_iterations:
                colors_used.append(len(np.unique(result)))
                j+=1
            break
        else:
            numero_final = use_less
            for k in range(ite_number):
                colors_used.append(use_less)
            
            first_coloring = False

	
    return result,len(set(result.values())),colors_used,numero_final


# In[ ]:





# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=12b3972d-d948-4710-91e7-f320364f32a3' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
