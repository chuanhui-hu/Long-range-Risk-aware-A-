#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
import pandas as pd
import cv2
from os import walk
import numpy as np
import matplotlib.pyplot as plt
import time
import heapq
from datetime import timedelta
from sklearn.cluster import AgglomerativeClustering
import multiprocessing
import random


test_count = int(sys.argv[1])
date = int(sys.argv[2])
SOG_threshold = int(sys.argv[3])

if date < 10:
    save_path = "results/07_0" + str(date) + "/"
else:
    save_path = "results/07_" + str(date) + "/"

save_path += str(test_count) + "/"

if SOG_threshold >= 20:
    save_path += "without_prediction/"
else:
    save_path += "with_prediction/"

print("running test: ", test_count, date, SOG_threshold)
random.seed(test_count)
np.random.seed(test_count)


# In[2]:


class Vertex():  # for PRM
    def __init__(self, idx, x, y):
        self.idx = idx
        self.x = x
        self.y = y

class Node():  # for tree
    def __init__(self, idx, x, y, parent_idx=None):
        self.idx = idx
        self.x = x
        self.y = y
        self.parent_idx = parent_idx
        

def not_connected(V, E, vertex):
    V_num = len(V.keys())
    connected_set = set()
    queue = [vertex]
    while queue:
        idx = queue.pop()
        connected_set.add(idx)
        for neighbor_idx in E[idx]:
            if neighbor_idx not in connected_set:
                queue.append(neighbor_idx)
    
    return connected_set ^ set(V.keys())
                
def path_good(chart, pos0, pos1, increment):
    x0, y0 = pos0
    x1, y1 = pos1
    dist = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    direction = [(x1-x0)/dist, (y1-y0)/dist]
    for i in range(1, int(dist//increment), increment):
        if chart[int(y0+direction[1]*i*increment)][int(x0+direction[0]*i*increment)][0] > 0:
            return False
    
    return True

def geo_to_pixel(chart, geo_position, extent):
    lon_0, lon_1, lat_0, lat_1 = extent
    
    x_geo = geo_position[0]
    y_geo = geo_position[1]
    x_geo_range = lon_1 - lon_0
    y_geo_range = lat_1 - lat_0
    
    x_pixel = int((x_geo - lon_0) / x_geo_range * chart.shape[1])
    y_pixel = int((lat_1 - y_geo) / y_geo_range * chart.shape[0])
    
    pos_in_pixel = (x_pixel, y_pixel)
    return pos_in_pixel

def get_phi(x0, y0, x1, y1, l_local):
    x_disp = (x1 - x0)
    if x_disp > l_local:
        x_disp = l_local
    if x_disp < - l_local:
        x_disp = - l_local

    if y1 - y0 > 0:
        phi = np.arccos(x_disp / l_local)
    else:
        phi = 2*np.pi - np.arccos(x_disp / l_local)
    return phi

def draw_path(chart, path, color=None, width=None):
    if not color:
        color = (0, 255, 0)
    if not width:
        width = 5
    for i in range(len(path) - 1):
        start_point = path[i]
        end_point = path[i+1]
        cv2.line(chart, start_point, end_point, color, width)

def draw_ship(chart, pos, course_in_degree, scale, color):
    x, y = pos
    phi = (course_in_degree - 90) * np.pi / 180
    ship_contour = scale * np.matrix([[30, 0], 
                                      [10, -10], 
                                      [-30, -10], 
                                      [-30, 10], 
                                      [10, 10]]).T
    rot = np.matrix([[np.cos(phi), -np.sin(phi)], 
                     [np.sin(phi), np.cos(phi)]])
    ship_contour = np.round(np.matmul(rot, ship_contour)).astype(int)
    
    for i in range(5):
        ship_contour[0, i] += x
        ship_contour[1, i] += y
        
    cv2.fillPoly(chart, pts=[ship_contour.T], color=color)
    
    return chart

def draw_square(chart, pos, scale, color):
    x, y = pos
    ship_contour = scale * np.matrix([[20, -20], 
                                      [-20, -20], 
                                      [-20, 20], 
                                      [20, 20]]).T
    
    for i in range(4):
        ship_contour[0, i] += x
        ship_contour[1, i] += y
        
    cv2.fillPoly(chart, pts=[ship_contour.T], color=color)
    
    return chart


# In[3]:


directory = "../nautical_chart/"
df_description = pd.read_csv(directory+"chart_detail.csv")
# print(df_description)

resolution, x_km_per_deg, y_km_per_deg, lon_0, lon_1, lat_0, lat_1 = df_description.loc[0, :]
extent = (lon_0, lon_1, lat_0, lat_1)


# # good test cases (21:00)
# 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31
# 
# 4: good: 21:56:57 to 22:59:56 <br>
# 5: simple: 21:03:23 to 22:46:24 <br>
# 6: good: 21:49:05 to 22:23:13 <br>
# 12: following another ship: 21:00:25 to 22:29:55 <br>
# 19: good: 22:06:13 to 22:52:11 <br>
# 25: good: 22:12:47 to 22:50:18 <br>

# In[4]:


# read test cases
test_cases = {4: ["21:56:57", "22:59:56"], 
              5: ["21:03:23", "22:46:24"], 
              6: ["21:49:05", "22:23:13"], 
              12: ["21:00:25", "22:29:55"], 
              19: ["22:06:13", "22:52:11"], 
              25: ["22:12:47", "22:50:18"]}
# date = 4
begin = test_cases[date][0]
end = test_cases[date][1]

# read test cases
test_case_path = "../test_cases/"
df_test_cases = pd.read_csv(test_case_path + "test_cases.csv")
own_mmsi = int(df_test_cases.loc[date-1, "own_mmsi"])
if date < 10:
    filename = "AIS_2020_07_0" + str(date) + ".csv"
else:
    filename = "AIS_2020_07_" + str(date) + ".csv"
csv_directory = "../processed_data/"
df = pd.read_csv(csv_directory + filename)


# In[5]:


# collect the latest ship positions (SOG < 0.5) in the past 10 minutes

chart_path = "../nautical_chart/res_100_lon_-122.67_-122.22_lat_37.54_38.17.jpg"
chart = cv2.imread(chart_path)

df = pd.read_csv(csv_directory + filename)

t_end = pd.to_datetime('2020-07-' + filename[-6:-4] + ' ' + begin)
t_start = t_end - timedelta(minutes=10)

ship_dict = {}
for i in range(df.shape[0]):
    MMSI, BaseDateTime, LAT, LON, SOG, COG, Heading = df.loc[i, :]
    if t_start <= pd.to_datetime(BaseDateTime) <= t_end:
        if MMSI not in ship_dict:
            ship_dict[MMSI] = []
        ship_dict[MMSI].append([BaseDateTime, LAT, LON, SOG, COG])

for key in ship_dict:
    ship_dict[key].sort(key=lambda x : x[0])


ships_10min = {}
for mmsi in ship_dict:
    for (BaseDateTime, LAT, LON, SOG, COG) in ship_dict[mmsi]:
        if SOG < 0.5:
            if mmsi not in ships_10min:
                ships_10min[mmsi] = []
            pos = geo_to_pixel(chart, [LON, LAT], extent)
            ships_10min[mmsi] = [pos[0], pos[1], COG]

to_be_clustered = []
for mmsi in ships_10min:
    to_be_clustered.append(ships_10min[mmsi][:2])


# In[6]:


# perform hierarchical clustering and display all the clusters by color

display_each = False
# n_clusters = 20
# clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="complete").fit(to_be_clustered)
clustering = AgglomerativeClustering(n_clusters=None, linkage="complete", distance_threshold=400, compute_full_tree=True).fit(to_be_clustered)


ship_mmsi_list = list(ships_10min.keys())
cm = plt.get_cmap('gist_rainbow')

if display_each:
    for target_label in range(clustering.n_clusters_):
        image = chart.copy()
        for i in range(len(ship_mmsi_list)):
            mmsi = ship_mmsi_list[i]
            pos = ships_10min[mmsi][:2]
            COG = ships_10min[mmsi][2]
            label = clustering.labels_[i]
            if label != target_label:
                continue
            color = [int(x*255) for x in cm(label/n_clusters)[:3]]

            image = draw_ship(image, pos, COG, scale=3, color=color)

#         image = cv2.resize(image, [image.shape[1] // 10, image.shape[0] // 10])
#         cv2.imshow("image", image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

else:
    image = chart.copy()
    for i in range(len(ship_mmsi_list)):
        mmsi = ship_mmsi_list[i]
        pos = ships_10min[mmsi][:2]
        COG = ships_10min[mmsi][2]
        label = clustering.labels_[i]
        color = [int(x*255) for x in cm(label/clustering.n_clusters_)[:3]]

        image = draw_ship(image, pos, COG, scale=3, color=color)
        cv2.imwrite(save_path + "clustering/" + "clusters.jpg", image)
#     image = cv2.resize(image, [image.shape[1] // 10, image.shape[0] // 10])
#     cv2.imshow("image", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# In[7]:


# get the median position of each cluster
clusters = {}
for i in range(len(ship_mmsi_list)):
    mmsi = ship_mmsi_list[i]
    label = clustering.labels_[i]
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(ships_10min[mmsi][0:2])

cluster_medians = []
for label in range(clustering.n_clusters_):
    mean = [0, 0]
    for pos in clusters[label]:
        mean[0] += pos[0]
        mean[1] += pos[1]
    mean[0] /= len(clusters[label])
    mean[1] /= len(clusters[label])
    
    median = []
    min_dist_square = 1000000
    for pos in clusters[label]: 
        if (pos[0] - mean[0])**2 + (pos[1] - mean[1])**2 < min_dist_square:
            min_dist_square = (pos[0] - mean[0])**2 + (pos[1] - mean[1])**2
            median = pos.copy()
    cluster_medians.append(median)

image = chart.copy()
for i in range(len(ship_mmsi_list)):
    mmsi = ship_mmsi_list[i]
    pos = ships_10min[mmsi][:2]
    COG = ships_10min[mmsi][2]
    label = clustering.labels_[i]
    color = [int(x*255) for x in cm(label/clustering.n_clusters_)[:3]]
    if pos in cluster_medians:
#         color = [0, 0, 255]
        image = draw_ship(image, pos, COG, scale=3, color=color)
        cv2.imwrite(save_path + "clustering/" + "cluster_medians.jpg", image)
# image = cv2.resize(image, [image.shape[1] // 10, image.shape[0] // 10])
# cv2.imshow("image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print("cluster medians: ", cluster_medians)


# In[8]:


# use PRM to generate a road map
chart_path = "../nautical_chart/res_100_lon_-122.67_-122.22_lat_37.54_38.17.jpg"
chart = cv2.imread(chart_path)

# PRM_save_path = "../draft/figures/generated_PRM/"

V = {}  # idx: Vertex()
E = {}  # idx: [neighbor_idx]
V_count = 0
max_V_num = 15000
max_neighbor_num = 16
max_course_change = np.pi/4  # rad

# initialize vertices for targets
targets_in_ocean = [[50, 3200], [50, 5000], [50, 6800], [1200, 6800], [3800, 6500], [3800, 5300], [3950, 1200], [3300, 100]]

# targets = [[1641, 3443], [3132, 3678], [2713, 4017], [2679, 2837], [2386, 4087], [3353, 4218], [1865, 3692], [3078, 3960], [2548, 4687], [2495, 4324], [2953, 4590], [1500, 2210], [1987, 4047], [2275, 2754], [1721, 2902], [1189, 6926], [3756, 1233], [3522, 766], [3562, 4288], [3093, 6450], [2202, 3479], [2921, 4809]]
targets = cluster_medians + targets_in_ocean
for target in targets:  
    # if the ship position is not in the water on the chart (due to chart accuracy)
    if chart[target[1]][target[0]][0] > 0:  
        while True:
            dx = int(np.random.rand() * 100 - 50)
            dy = int(np.random.rand() * 100 - 50)
            if chart[target[1] + dy][target[0] + dx][0] == 0:
                target[0] += dx
                target[1] += dy
                break

    V[V_count] = Vertex(V_count, target[0], target[1])
    V_count += 1

cpu_time = time.time()
        
# generate PRM
while V_count < max_V_num:
    x = int(np.random.rand() * chart.shape[1])
    y = int(np.random.rand() * chart.shape[0])
    if chart[y][x][0] > 0:
        continue
    V[V_count] = Vertex(V_count, x, y)
    V_count += 1

for i in range(V_count):
    E[i] = set()

for vertex in range(V_count):
    if len(E[vertex]) >= max_neighbor_num:
        continue
    dist_heap = []
    x0, y0 = V[vertex].x, V[vertex].y
    for neighbor in range(V_count):
        x1, y1 = V[neighbor].x, V[neighbor].y
        dist = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        if dist < 1:  # if at the same pixel position
            continue
        
        if len(dist_heap) < max_neighbor_num:
            heapq.heappush(dist_heap, [-dist, neighbor])
        else:
            heapq.heappushpop(dist_heap, [-dist, neighbor])
    
    while len(E[vertex]) < max_neighbor_num and dist_heap:
        neg_dist, neighbor = heapq.heappop(dist_heap)
        x1, y1 = V[neighbor].x, V[neighbor].y
        if path_good(chart, [x0, y0], [x1, y1], 2):
            E[vertex].add(neighbor)
            E[neighbor].add(vertex)

        
# print("PRM takes time: ", time.time() - cpu_time)
with open(save_path + "log/" + "log.txt", 'a') as f:
    f.write("trees take time: " + str(time.time() - cpu_time) + "\n")

# In[9]:


# save the generated roadmap
image = chart.copy()

for vertex in range(V_count):
    x0, y0 = V[vertex].x, V[vertex].y
    for neighbor in E[vertex]:
        x1, y1 = V[neighbor].x, V[neighbor].y
        draw_path(image, [[x0, y0], [x1, y1]], color=[0, 0, 255])

# cv2.imwrite(PRM_save_path + str(date) + "/PRM_" + str(max_V_num) + "_" + str(max_neighbor_num) + ".jpg", image)
cv2.imwrite(save_path + "PRM_tree/" + "PRM_" + str(max_V_num) + "_" + str(max_neighbor_num) + ".jpg", image)

# In[10]:


# from the road map, generate a tree for each target by Dijkstra's algorithm (modified cost function)

coef0 = 1000
coef1 = 0.1
p_threshold = 1000000  # not activated

def get_cost(dist_sum, p_sum, p_max):
    dist_cost = dist_sum
    turn_cost = (p_max + coef1 * p_sum)
    return dist_cost + coef0 * turn_cost

cpu_time = time.time()
tree_list = []  # list of node_dict
for t_idx in range(len(targets)):
    if len(not_connected(V, E, t_idx)) > 0.5 * V_count:  # if the target is not connected, find a nearby one as the new target
        x0, y0 = V[t_idx].x, V[t_idx].y
        dist_heap = []
        for i in range(V_count):
            x1, y1 = V[i].x, V[i].y
            heapq.heappush(dist_heap, [np.sqrt((x1-x0)**2+(y1-y0)**2), i])
        while dist_heap:
            dist, idx = heapq.heappop(dist_heap)
            if len(not_connected(V, E, idx)) < 0.1 * V_count:
                target = [V[idx].x, V[idx].y]
                break
    else:
        target = targets[t_idx]
        idx = t_idx
        
    node_heap = [[0, 0, 0, 0, idx, None, None, None]]  # heap of [cost, dist_sum, p_sum, p_max, vertex_idx, parent_idx, phi, l] 
    node_dict = {}  # {node_idx: Node()}
    visited = set()
    
    while node_heap:
        cost, dist_sum, p_sum, p_max, vertex_idx, parent_idx, phi_parent, l_parent = heapq.heappop(node_heap)
        if vertex_idx in visited:
            continue
        visited.add(vertex_idx)
        x0, y0 = V[vertex_idx].x, V[vertex_idx].y
        node_dict[vertex_idx] = Node(vertex_idx, x0, y0, parent_idx)
        
        for neighbor in E[vertex_idx]:
            x1, y1 = V[neighbor].x, V[neighbor].y
            l_local = np.sqrt((x1-x0)**2+(y1-y0)**2)  # edge length
            phi = get_phi(x0, y0, x1, y1, l_local)
            
            # calculate the cost function
            p_local = 0
            if phi_parent != None:
                p_local = abs(np.tan(0.5 * (phi - phi_parent))) / min(l_local, l_parent)
            
            # do not allow sharp turn
            if p_local > p_threshold:
                continue
            
            cost = get_cost(dist_sum+l_local, p_sum+p_local, max(p_max,p_local))
            heapq.heappush(node_heap, [cost, dist_sum+l_local, p_sum+p_local, max(p_max,p_local), neighbor, vertex_idx, phi, l_local])
    
    tree_list.append(node_dict)
    
    image = chart.copy()

    for vertex in node_dict:
        x0, y0 = node_dict[vertex].x, node_dict[vertex].y
        if node_dict[vertex].parent_idx != None:
            parent = node_dict[vertex].parent_idx
            x1, y1 = node_dict[parent].x, node_dict[parent].y
            draw_path(image, [[x0, y0], [x1, y1]], color=[0, 0, 255])
    
    image = draw_square(image, target, scale=2, color=[0, 255, 0])

    # cv2.imwrite(PRM_save_path + str(date) + "/PRM_tree_" + str(idx) + "_" + str(coef0) + "_" + str(coef1) + ".jpg", image)
    cv2.imwrite(save_path + "PRM_tree/" + "/PRM_tree_" + str(idx) + "_" + str(coef0) + "_" + str(coef1) + ".jpg", image)

# print("trees take time: ", time.time() - cpu_time)
with open(save_path + "log/" + "log.txt", 'a') as f:
    f.write("trees take time: " + str(time.time() - cpu_time) + "\n")


# In[11]:


def get_path(pos, COG, node_dict, V):
    own_phi = COG*np.pi/180 - np.pi/2
    path = [[pos[0], pos[1]]]
    min_cost = 100000000
    closest_node_idx = None
    for i in node_dict:
        x, y = node_dict[i].x, node_dict[i].y
        dist = np.sqrt((x-pos[0])**2 + (y-pos[1])**2)
        phi = get_phi(pos[0], pos[1], x, y, dist)
        p_local = abs(np.tan(0.5 * (phi - own_phi))) # / l_local
        cost = get_cost(dist, p_local, p_local)
        if cost < min_cost:
            min_cost = cost
            closest_node_idx = i
    
    cur_node = node_dict[closest_node_idx]
    x, y = cur_node.x, cur_node.y
    path.append([x, y])
    while cur_node.parent_idx != None:
        cur_node = node_dict[cur_node.parent_idx]
        x, y = cur_node.x, cur_node.y
        path.append([x, y])
    return path

def get_weight(pos, COG, path, dist_threshold=None):
    course = COG * np.pi / 180 - np.pi/2  # transfer the COG to the image coordinate
    if not dist_threshold:
        dist_threshold = 5000
    dist_sum = 0
    for i in range(1, len(path)):
        x0, y0 = path[i-1]
        x1, y1 = path[i]
        dist_sum += np.sqrt((x1-x0)**2 + (y1-y0)**2)
        if dist_sum > dist_threshold:
            break
    
    x, y = pos
    xd, yd = path[i]
    dist_d = np.sqrt((xd-x)**2 + (yd-y)**2)
    
#     if dist_d == 0:
#         return 2
    
#     direction = [(xd-x)/dist_d, (yd-y)/dist_d]
    phi = get_phi(x, y, xd, yd, dist_d)
    
    return 7 * np.cos(phi-course)
    


# In[12]:


# # given a position, find a path to each possible destination

# save_path = "../draft/figures/find_path_test/"
# display_all = True

# pos = [2500, 3300]
# COG = 180  # to the south
# path_list = []
# weight_list = []

# # t0 = time.time()

# for tree in tree_list:
#     ship_path = get_path(pos, COG, tree, V)
#     path_list.append(ship_path)
#     weight_list.append(get_weight(pos, COG, ship_path))

# if display_all:
#     weight_sum = sum([np.exp(w) for w in weight_list])
#     image = chart.copy()
#     for i in range(len(path_list)):
#         ship_path = path_list[i]
        
#         width = int(200 * np.exp(weight_list[i]) / weight_sum)
#         if width:
#             draw_path(image, ship_path, width=width)

# #     draw_ship(image, pos, COG, scale=3, color=[0, 0, 255])
#     cv2.imwrite(save_path + "find_path_" + str(pos[0]) + "_" + str(pos[1]) + \
#                 "_to_all.jpg", image)

#     image = cv2.resize(image, [image.shape[1] // 10, image.shape[0] // 10])
#     cv2.imshow("image", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
        
    
# else:
#     for i in range(len(path_list)):
#         ship_path = path_list[i]
#         image = chart.copy()
#         draw_path(image, ship_path)

#         cv2.imwrite(save_path + "find_path_" + str(pos[0]) + "_" + str(pos[1]) + "_" + str(COG) +\
#                     "_to_" + str(ship_path[-1][0]) + "_" + str(ship_path[-1][1]) + ".jpg", image)

# # print("time consumed to find possible paths for one ship: ", time.time() - t0)        


# In[13]:


def get_cost_with_risk(dist_sum, p_sum, p_max, risk_sum):
    dist_cost = dist_sum
    turn_cost = (p_max + coef1 * p_sum)
    risk_cost = risk_sum
    return dist_cost + coef0 * turn_cost + coef2 * risk_cost

def get_pos_prediction(path, arrival_time, path_start_time, cur_time):
    t = (cur_time - path_start_time).seconds
    if t >= arrival_time[-1]:
        return [path[-1][0], path[-1][1]]
    
    i, j = 0, len(arrival_time) - 1
    while i < j:
        mid = (i+j) // 2
        if arrival_time[mid] < t:
            i = mid + 1
        else:
            j = mid
    
    x0, y0 = path[i-1]
    x1, y1 = path[i]
    x = x0 + (x1-x0) * (t - arrival_time[i-1]) / (arrival_time[i] - arrival_time[i-1])
    y = y0 + (y1-y0) * (t - arrival_time[i-1]) / (arrival_time[i] - arrival_time[i-1])
    return [x, y]
    
    


# In[22]:


# path planning of the own ship on test cases

CORE_NUM = 16

# read chart
chart_path = "../nautical_chart/res_100_lon_-122.67_-122.22_lat_37.54_38.17.jpg"
chart = cv2.imread(chart_path)

coef0 = 1000
coef1 = 0.1
coef2 = 20000
risky_dist_threshold = 500  # in pixel

# read test cases
test_case_path = "../test_cases/"
df_test_cases = pd.read_csv(test_case_path + "test_cases.csv")
own_mmsi = int(df_test_cases.loc[date-1, "own_mmsi"])
if date < 10:
    filename = "AIS_2020_07_0" + str(date) + ".csv"
else:
    filename = "AIS_2020_07_" + str(date) + ".csv"
csv_directory = "../processed_data/"
df = pd.read_csv(csv_directory + filename)


# define start and end time, and read AIS data in this time period
start_time = pd.to_datetime('2020-07-' + filename[-6:-4] + ' ' + begin) 
end_time = pd.to_datetime('2020-07-' + filename[-6:-4] + ' ' + end)

ship_dict = {}  # ship_dict[MMSI]: list of [BaseDateTime, LAT, LON, SOG, COG]
for i in range(df.shape[0]):
    MMSI, BaseDateTime, LAT, LON, SOG, COG, Heading = df.loc[i, :]
    if MMSI == own_mmsi and (pd.to_datetime(BaseDateTime) < start_time or pd.to_datetime(BaseDateTime) > end_time):
        continue
    if start_time - timedelta(hours=1) <= pd.to_datetime(BaseDateTime) <= end_time + timedelta(hours=1):
        if MMSI not in ship_dict:
            ship_dict[MMSI] = []
        ship_dict[MMSI].append([BaseDateTime, LAT, LON, SOG, COG])
for key in ship_dict:
    ship_dict[key].sort(key=lambda x : x[0])

# print(ship_dict[own_mmsi])

# get the start position, SOG, COG, and goal position
start_pos_geo = ship_dict[own_mmsi][0][2], ship_dict[own_mmsi][0][1]
goal_pos_geo = ship_dict[own_mmsi][-1][2], ship_dict[own_mmsi][-1][1]
start_pos = geo_to_pixel(chart, start_pos_geo, extent)
start_SOG = ship_dict[own_mmsi][0][3]
start_COG = ship_dict[own_mmsi][0][4]
goal_pos = geo_to_pixel(chart, goal_pos_geo, extent)

# find a closest vertex to serve as the goal vertex on the PRM
closest_idx = None
start_vertex_heap = []
for i in range(V_count):
    x, y = V[i].x, V[i].y
    dist = np.sqrt((goal_pos[0]-x)**2 + (goal_pos[1]-y)**2)
    heapq.heappush(start_vertex_heap, [dist, i])
while start_vertex_heap:
    closest_dist, closest_idx = heapq.heappop(start_vertex_heap)
    if len(not_connected(V, E, closest_idx)) < 0.1 * V_count:
        break
goal_pos = [V[closest_idx].x, V[closest_idx].y]
    
# start from the goal vertex and generate a tree, calculate the heuristic cost for A*
# heap of [cost, dist_sum, p_sum, p_max, vertex_idx, parent_idx, phi, l] 
node_heap = [[0, 0, 0, 0, closest_idx, None, None, None]]  
goal_tree = {}  # {node_idx: Node()}
cost_h_to_goal = {}
visited = set()

while node_heap:
    cost, dist_sum, p_sum, p_max, vertex_idx, parent_idx, phi_parent, l_parent = heapq.heappop(node_heap)
    if vertex_idx in visited:
        continue
    visited.add(vertex_idx)
    x0, y0 = V[vertex_idx].x, V[vertex_idx].y
    goal_tree[vertex_idx] = Node(vertex_idx, x0, y0, parent_idx)
    cost_h_to_goal[vertex_idx] = cost

    for neighbor in E[vertex_idx]:
        x1, y1 = V[neighbor].x, V[neighbor].y
        l_local = np.sqrt((x1-x0)**2+(y1-y0)**2)  # edge length
        phi = get_phi(x0, y0, x1, y1, l_local)

        # calculate the cost function
        p_local = 0
        if phi_parent != None:
            p_local = abs(np.tan(0.5 * (phi - phi_parent))) # / min(l_local, l_parent)

        cost = get_cost(dist_sum+l_local, p_sum+p_local, max(p_max,p_local))
        heapq.heappush(node_heap, [cost, dist_sum+l_local, p_sum+p_local, max(p_max,p_local), neighbor, vertex_idx, phi, l_local])

    
# display current time on the image    
WHITE = (255,255,255)
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 3
font_color = WHITE
font_thickness = 10

time_step = 120  # in minutes
# SOG_threshold = 2  # fast ships: SOG >= SOG_threshold
# risk_threshold = 5e-3
initial_sigma = 10
a_sigma = 0.02

own_pos = start_pos
own_COG = start_COG
own_SOG = start_SOG

# print("own_SOG: ", own_SOG)

own_path = [[[start_pos[0], start_pos[1]]]]  # list of former path
own_planned_path = []  # list of newly planned path

t = start_time
while t < end_time + timedelta(hours=1):
    x0, y0 = own_pos
    xg, yg = goal_pos
    
    
    # get the latest observation of target ships
    latest_record = {}
    for mmsi in ship_dict:
        if mmsi == own_mmsi:
            continue
        while len(ship_dict[mmsi]) > 1 and pd.to_datetime(ship_dict[mmsi][1][0]) <= t:
            ship_dict[mmsi].pop(0)
        
        BaseDateTime, LAT, LON, SOG, COG = ship_dict[mmsi][0]
        latest_record[mmsi] = [BaseDateTime, LAT, LON, SOG, COG]
    
    if np.sqrt((xg-x0)**2 + (yg-y0)**2) < 10:  # if have reached the goal
        break
    
    cpu_time = time.time()
    possible_path = {}
    
#     for mmsi in latest_record:
    def find_possible_path(mmsi):
        BaseDateTime, LAT, LON, SOG, COG = latest_record[mmsi]
        possible_path = {}
        if SOG < SOG_threshold:  # slow ships
            if mmsi in possible_path:
                possible_path.pop(mmsi)
#             continue
            return None
        
        if mmsi not in possible_path:
            possible_path[mmsi] = {}
        if possible_path[mmsi] == {} or possible_path[mmsi]["time"] < BaseDateTime:  # if need initialization or need update
            possible_path[mmsi]["time"] = BaseDateTime

            pos = geo_to_pixel(chart, [LON, LAT], extent)
            possible_path[mmsi]["pos"] = pos
            possible_path[mmsi]["SOG"] = SOG
            possible_path[mmsi]["COG"] = COG

            possible_path[mmsi]["path_list"] = []
            possible_path[mmsi]["arrival_time_list"] = []
            possible_path[mmsi]["weight_list"] = []
            for tree in tree_list:  # the root of the tree is the position of the possible destination
                ship_path = get_path(pos, COG, tree, V)
                possible_path[mmsi]["path_list"].append(ship_path)
                arrival_time_list = [0]
                for i in range(1, len(ship_path)):
                    x0, y0 = ship_path[i-1]
                    x1, y1 = ship_path[i]
                    edge_dist = np.sqrt((x1-x0)**2 + (y1-y0)**2)
                    arrival_time = arrival_time_list[i-1] + edge_dist / (SOG * 0.514444 * resolution / 1000)
                    arrival_time_list.append(arrival_time)
                possible_path[mmsi]["arrival_time_list"].append(arrival_time_list)
                possible_path[mmsi]["weight_list"].append(get_weight(pos, COG, ship_path))

            weight_sum = sum([np.exp(w) for w in possible_path[mmsi]["weight_list"]])
            for i, w in enumerate(possible_path[mmsi]["weight_list"]):
                possible_path[mmsi]["weight_list"][i] = np.exp(w) / weight_sum
                
#             # produce a interpolated path list, time step = 1
#             possible_path[mmsi]["interpolated_path_list"] = []
#             for i, path in enumerate(possible_path[mmsi]["path_list"]):
#                 interpolated_path = []
#                 speed = possible_path[mmsi]["SOG"] * 0.514444 * resolution / 1000
#                 pos_idx = 0
#                 for interpolated_t in range(int(possible_path[mmsi]["arrival_time_list"][i][-1]) + 1):
#                     if interpolated_t > possible_path[mmsi]["arrival_time_list"][i][pos_idx]:
#                         pos_idx += 1
#                     if pos_idx + 1 >= len(possible_path[mmsi]["arrival_time_list"][i]):
#                         break
#                     t0 = possible_path[mmsi]["arrival_time_list"][i][pos_idx]
#                     t1 = possible_path[mmsi]["arrival_time_list"][i][pos_idx+1]
#                     x0, y0 = path[pos_idx]
#                     x1, y1 = path[pos_idx+1]
#                     x = x0 + (x1-x0) * (interpolated_t - t0) / (t1 - t0)
#                     y = y0 + (y1-y0) * (interpolated_t - t0) / (t1 - t0)
#                     interpolated_path.append([x, y])
                    
#                 if len(interpolated_path) < 1:
#                     interpolated_path.append(path[-1])
#                 possible_path[mmsi]["interpolated_path_list"].append(interpolated_path)
        return possible_path
        
    pool = multiprocessing.Pool(CORE_NUM)
    results = pool.map(func=find_possible_path, iterable=list(latest_record.keys()))
    pool.close()
    pool.join()
    
    for result in results:
        if result:
            for mmsi in result:
                possible_path[mmsi] = result[mmsi]
                
    # print("time consumed on finding possible path for target ships: ", time.time() - cpu_time)
    with open(save_path + "log/" + "log.txt", 'a') as f:
        f.write('time consumed on finding possible path for target ships: ' + str(time.time() - cpu_time) + "\n")

    cpu_time = time.time()
    
    # plan the path of the own ship based on the cost(dist_cost, turn_cost, risk_cost)
    x0, y0 = own_pos  # x, y in pixel
    own_phi = own_COG*np.pi/180 - np.pi/2  # in rad
    own_speed = own_SOG * 0.514444 * resolution / 1000 # in pixels/s
    
    # find a vertex on the PRM to be the first point to connect by the own path
    own_start_idx = None
    dist_heap = []
    for i in range(V_count):
        x1, y1 = V[i].x, V[i].y
        l_local = np.sqrt((x1-x0)**2 + (y1-y0)**2)
        phi = get_phi(x0, y0, x1, y1, l_local)
#         p_local = abs(np.tan(0.5 * (phi - own_phi))) / l_local
        p_local = abs(np.tan(0.5 * (phi - own_phi))) 
        cost = l_local + coef0 * (p_local + coef1 * p_local)
        heapq.heappush(dist_heap, [cost, i])
    while dist_heap:
        cost, i = heapq.heappop(dist_heap)
        if len(not_connected(V, E, i)) < 0.1 * max_V_num:
            own_start_idx = i
            min_cost = cost
            break
    
    x1, y1 = V[own_start_idx].x, V[own_start_idx].y
    l_local = np.sqrt((x1-x0)**2 + (y1-y0)**2)
    phi = get_phi(x0, y0, x1, y1, l_local)
    p_local = abs(np.tan(0.5 * (phi - own_phi))) # / l_local
    own_time = l_local / own_speed
    
    # heap of [cost_f, cost_g, dist_sum, p_sum, p_max, risk_sum, own_time, vertex_idx, [parent_idx], phi, l] 
    visited = set()
    node_heap = [[min_cost, min_cost, l_local, p_local, p_local, 0, own_time, own_start_idx, [], phi, l_local]]
    while node_heap:
#         print(len(visited))
        cost_f, cost_g, dist_sum, p_sum, p_max, risk_sum, own_time, vertex_idx, parent_idx_list, phi_parent, l_parent = heapq.heappop(node_heap)
        if vertex_idx in visited:
            continue
        visited.add(vertex_idx)
        x0, y0 = V[vertex_idx].x, V[vertex_idx].y
        # print(parent_idx_list, vertex_idx)
        xg, yg = goal_pos
        if np.sqrt((xg-x0)**2 + (yg-y0)**2) < 10:
#             print("found path: ", parent_idx_list)
            break
        
#         print("current node time: ", own_time, ", dist to goal: ", np.sqrt((xg-x0)**2 + (yg-y0)**2))
        
        
        # run the cost of neighbors parallelly
        # for neighbor in E[vertex_idx]:
        neighbors = E[vertex_idx]
        def f(neighbor):
            if neighbor in visited:
                return []
            x1, y1 = V[neighbor].x, V[neighbor].y
            l_local = np.sqrt((x1-x0)**2 + (y1-y0)**2)  # edge length
            phi = get_phi(x0, y0, x1, y1, l_local)
#             cost_h = np.sqrt((xg-x1)**2 + (yg-y1)**2)  # heuristic cost
            cost_h = cost_h_to_goal[neighbor]
            
            # calculate the turn cost
            p_local = abs(np.tan(0.5 * (phi - phi_parent))) # / min(l_local, l_parent)
            
            new_time_list = []
            new_pos_list = []
            for i in range(1, int(l_local / own_speed // 300)):
                delta_t = 300 * i
                new_time_list.append(own_time + delta_t)
                new_pos_list.append([x0 + own_speed*delta_t*np.cos(phi), y0 + own_speed*delta_t*np.sin(phi)])
            
            new_time_list.append(own_time + l_local / own_speed)
            new_pos_list.append([V[neighbor].x, V[neighbor].y])
                
#             new_time = own_time + l_local / own_speed
            
            # calculate the risk cost at each intermedia point
            risk_neighbor = 0
            for i in range(len(new_pos_list)):
                x1, y1 = new_pos_list[i]
                new_time = new_time_list[i]
                
                
                for mmsi in latest_record:
                    BaseDateTime, LAT, LON, SOG, COG = latest_record[mmsi]
                    if SOG <= SOG_threshold:  # slow ship, predict the next position by its current pos, COG, SOG
                        x_target, y_target = geo_to_pixel(chart, [LON, LAT], extent)
                        dt_target = (t + timedelta(seconds=new_time) - pd.to_datetime(BaseDateTime)).seconds
                        sigma = initial_sigma + a_sigma * dt_target
                        
                        speed_target = SOG * 0.514444 * resolution / 1000
                        x_target += np.cos(COG*np.pi/180 - np.pi/2) * speed_target * dt_target
                        y_target += np.sin(COG*np.pi/180 - np.pi/2) * speed_target * dt_target

                        dist_target = np.sqrt((x_target-x1)**2 + (y_target-y1)**2)
                        # if dist_target == 0:
                        #     return []
                        # if dist_target < risky_dist_threshold:
                        #     # risk_neighbor = 1 / dist_target
                        
                        risk_target = np.exp(-0.5*(dist_target/sigma)**2) / sigma
                        # if risk_target > risk_neighbor:
                        risk_neighbor += risk_target
                                
                    else:  # fast ship
                        path_list = possible_path[mmsi]["path_list"]
                        arrival_time_list = possible_path[mmsi]["arrival_time_list"]
                        weight_list = possible_path[mmsi]["weight_list"]
                        risk_target = 0
                        for i in range(len(path_list)):
                            path = path_list[i]
                            arrival_time = arrival_time_list[i]
                            weight = weight_list[i]

                            x_target, y_target =  get_pos_prediction(path, arrival_time, pd.to_datetime(BaseDateTime), t+timedelta(seconds=new_time))
                            dt_target = (t + timedelta(seconds=new_time) - pd.to_datetime(BaseDateTime)).seconds
                            sigma = initial_sigma + a_sigma * dt_target
    #                         time_from_last_record = (t - pd.to_datetime(BaseDateTime)).seconds + new_time
    #                         t0 = int(time_from_last_record)
    #                         t1 = t0 + 1
    #                         if 0 <= t0 and t1 < len(interpolated_path):
    #                             x0_target, y0_target = interpolated_path[t0]
    #                             x1_target, y1_target = interpolated_path[t1]
    #                             x_target = x0_target + (x1_target-x0_target) * (time_from_last_record - t0) / (t1 - t0)
    #                             y_target = y0_target + (y1_target-y0_target) * (time_from_last_record - t0) / (t1 - t0)
    #                         else:
    #                             x_target, y_target = interpolated_path[-1]
                            dist_target = np.sqrt((x_target-x1)**2 + (y_target-y1)**2)
                            # if dist_target == 0:
                            #     return []
                            # if dist_target < risky_dist_threshold:
                                # risk_target += weight / dist_target
                            risk_target += weight * np.exp(-0.5*(dist_target/sigma)**2) / sigma 
                        
                        # if risk_target > risk_neighbor:
                        #     risk_neighbor = risk_target
                        risk_neighbor += risk_target
                        
                    
                    # print(risk_neighbor)
                    # if risk_neighbor > risk_threshold:
                    #     break
                
#                 print(risk_neighbor)
                # if risk_neighbor > risk_threshold:
                #     return []
            
            cost_g = get_cost_with_risk(dist_sum+l_local, p_sum+p_local, max(p_max,p_local), risk_sum+risk_neighbor)
            return [cost_g+cost_h, cost_g, dist_sum+l_local, p_sum+p_local, max(p_max,p_local), risk_sum+risk_neighbor, new_time_list[-1], neighbor, parent_idx_list+[vertex_idx], phi, l_local]
        
        ################### end of f(neighbor) ###################
            
        pool = multiprocessing.Pool(CORE_NUM)
        results = pool.map(func=f, iterable=neighbors)
        pool.close()
        pool.join()
        
        for result in results:
            if result:
                heapq.heappush(node_heap, result)
                
                
                
                
#         for neighbor in E[vertex_idx]:
#             if neighbor in visited:
#                 continue
#             x1, y1 = V[neighbor].x, V[neighbor].y
#             l_local = np.sqrt((x1-x0)**2 + (y1-y0)**2)  # edge length
#             phi = get_phi(x0, y0, x1, y1, l_local)
# #             cost_h = np.sqrt((xg-x1)**2 + (yg-y1)**2)  # heuristic cost
#             cost_h = cost_h_to_goal[neighbor]
            
#             # calculate the turn cost
#             p_local = abs(np.tan(0.5 * (phi - phi_parent))) / min(l_local, l_parent)
#             new_time = own_time + l_local / own_speed
            
#             # calculate the risk cost
# #             tt = time.time()
#             risk_neighbor = 0
#             for mmsi in latest_record:
#                 BaseDateTime, LAT, LON, SOG, COG = latest_record[mmsi]
#                 if SOG <= SOG_threshold:  # slow ship, predict the next position by its current pos, COG, SOG
#                     x_target, y_target = geo_to_pixel(chart, [LON, LAT], extent)
#                     dt_target = (t + timedelta(seconds=new_time) - pd.to_datetime(BaseDateTime)).seconds
#                     speed_target = SOG * 0.514444 * resolution / 1000
#                     x_target += np.cos(COG*np.pi/180 - np.pi/2) * speed_target * dt_target
#                     y_target += np.sin(COG*np.pi/180 - np.pi/2) * speed_target * dt_target
                    
#                     dist_target = np.sqrt((x_target-x1)**2 + (y_target-y1)**2)
#                     if dist_target < risky_dist_threshold:
#                         if 1 / dist_target > risk_neighbor:
#                             risk_neighbor = 1 / dist_target
#                 else:  # fast ship
#                     interpolated_path_list = possible_path[mmsi]["interpolated_path_list"]
#                     weight_list = possible_path[mmsi]["weight_list"]
#                     for i in range(len(interpolated_path_list)):
#                         interpolated_path = interpolated_path_list[i]
#                         weight = weight_list[i]
#                         risk_target = 0
                        
#                         time_from_last_record = (t - pd.to_datetime(BaseDateTime)).seconds + new_time
#                         t0 = int(time_from_last_record)
#                         t1 = t0 + 1
#                         if 0 <= t0 and t1 < len(interpolated_path):
#                             x0_target, y0_target = interpolated_path[t0]
#                             x1_target, y1_target = interpolated_path[t1]
#                             x_target = x0_target + (x1_target-x0_target) * (time_from_last_record - t0) / (t1 - t0)
#                             y_target = y0_target + (y1_target-y0_target) * (time_from_last_record - t0) / (t1 - t0)
#                         else:
#                             x_target, y_target = interpolated_path[-1]
#                         dist_target = np.sqrt((x_target-x1)**2 + (y_target-y1)**2)
#                         if dist_target < risky_dist_threshold:
#                             risk_target += weight / dist_target
#                     if risk_target > risk_neighbor:
#                         risk_neighbor = risk_target
#                 if risk_neighbor > risk_threshold:
#                     break
            
#             if risk_neighbor > risk_threshold:
#                 continue
            
#             cost = get_cost(dist_sum+l_local, p_sum+p_local, max(p_max,p_local))
#             heapq.heappush(node_heap, [cost+cost_h, cost, dist_sum+l_local, p_sum+p_local, max(p_max,p_local), \
#                                        new_time, neighbor, parent_idx_list+[vertex_idx], phi, l_local])
        
                
#             heapq.heappush(node_heap, [cost+cost_h, cost, dist_sum+l_local, p_sum+p_local, max(p_max,p_local), \
#                                        new_time, neighbor, parent_idx_list+[vertex_idx], phi, l_local])
                                    # [cost_h, cost, dist_sum, p_sum, p_max, own_time, vertex_idx, [parent_idx], phi, l] 
#             print("push: ", neighbor, parent_idx_list+[vertex_idx])
            
    # print("time consumed on find the own path with lowest cost: ", time.time() - cpu_time)
    with open(save_path + "log/" + "log.txt", 'a') as f:
        f.write("time consumed on find the own path with lowest cost: " + str(time.time() - cpu_time) + "\n")
    
    # move the own ship forward and record the plan and path
    planned_path = [[own_pos[0], own_pos[1]]]
    for idx in parent_idx_list:
        planned_path.append([V[idx].x, V[idx].y])
    planned_path.append([V[vertex_idx].x, V[vertex_idx].y])
    own_planned_path.append(planned_path)    
    
            
    image = chart.copy()
    # draw target ships
    for mmsi in latest_record:
        BaseDateTime, LAT, LON, SOG, COG = latest_record[mmsi]
        pos_in_pixel = geo_to_pixel(image, [LON, LAT], extent)
        color = [0, 0, 255]
        draw_ship(image, pos_in_pixel, COG, scale=3, color=color)

    # draw own ship, path, and plan
    draw_ship(image, own_pos, own_COG, scale=3, color=[0, 255, 0])
    draw_path(image, own_path[-1], color=[0, 255, 0], width=10)
    draw_path(image, own_planned_path[-1], color=[0, 255, 255], width=10)

#     image = cv2.resize(image, [image.shape[1] // 10, image.shape[0] // 10])

    # show the current time on the image
    text = str(t)
    x, y = int(image.shape[1] * 0.05), int(image.shape[0] * 0.95)
    image = cv2.putText(image, text, (x,y), font, font_size, font_color, font_thickness, cv2.LINE_AA)

#     cv2.imshow("image", image)
#     cv2.waitKey(20)

    # test_save_path = "../draft/figures/test/"
    cv2.imwrite(save_path + "path/" + str(t) + ".jpg", image)
    
    # update the position of the own ship
    arrived = [[own_pos[0], own_pos[1]]]
    target_forward_dist = 60*time_step * own_speed  # pixel dist
    forward_dist = 0
    for i in range(len(planned_path) - 1):
        x0, y0 = planned_path[i]
        x1, y1 = planned_path[i+1]
        l_local = np.sqrt((x1-x0)**2+(y1-y0)**2)  # edge length
        phi = get_phi(x0, y0, x1, y1, l_local)
        if forward_dist + l_local < target_forward_dist:
            forward_dist += l_local
            x_reach = x1
            y_reach = y1
            arrived.append([int(x_reach), int(y_reach)])
        else:
            dist = target_forward_dist - forward_dist
            x_reach = x0 + (x1-x0) * dist / l_local
            y_reach = y0 + (y1-y0) * dist / l_local
            arrived.append([int(x_reach), int(y_reach)])
            break
    own_path.append(own_path[-1] + arrived)
    
    own_pos = [int(x_reach), int(y_reach)]
    own_COG = phi/np.pi * 180 + 90
    
    t += timedelta(minutes=time_step)


    
    
    
    
    
    
    
image = chart.copy()
# draw target ships
for mmsi in latest_record:
    BaseDateTime, LAT, LON, SOG, COG = latest_record[mmsi]
    pos_in_pixel = geo_to_pixel(image, [LON, LAT], extent)
    color = [0, 0, 255]
    draw_ship(image, pos_in_pixel, COG, scale=3, color=color)

# draw own ship, path, and plan
draw_ship(image, own_pos, own_COG, scale=3, color=[0, 255, 0])
draw_path(image, own_path[-1], color=[0, 255, 0], width=10)

#     image = cv2.resize(image, [image.shape[1] // 10, image.shape[0] // 10])

# show the current time on the image
text = str(t)
x, y = int(image.shape[1] * 0.05), int(image.shape[0] * 0.95)
image = cv2.putText(image, text, (x,y), font, font_size, font_color, font_thickness, cv2.LINE_AA)

#     cv2.imshow("image", image)
#     cv2.waitKey(20)

# test_save_path = "../draft/figures/test/"
cv2.imwrite(save_path + "path/" + str(t) + ".jpg", image)
    
    
    
    
    
    
    
# cv2.destroyAllWindows()


# In[23]:


# interpolate the path of all target ships
ship_dict = {}  # ship_dict[MMSI]: list of [BaseDateTime, LAT, LON, SOG, COG]
for i in range(df.shape[0]):
    MMSI, BaseDateTime, LAT, LON, SOG, COG, Heading = df.loc[i, :]
    if start_time - timedelta(hours=1) <= pd.to_datetime(BaseDateTime) <= end_time + timedelta(hours=1):
        if MMSI not in ship_dict:
            ship_dict[MMSI] = []
        ship_dict[MMSI].append([BaseDateTime, LAT, LON, SOG, COG])
for key in ship_dict:
    ship_dict[key].sort(key=lambda x : x[0])    

interpolated_path = {}
for mmsi in ship_dict:
    interpolated_path[mmsi] = {}
    for i in range(len(ship_dict[mmsi]) - 1):
        BaseDateTime, LAT, LON, SOG, COG = ship_dict[mmsi][i]
        t0 = (pd.to_datetime(BaseDateTime) - start_time).seconds + (pd.to_datetime(BaseDateTime) - start_time).days *24*3600
        x0, y0 = geo_to_pixel(chart, [LON, LAT], extent)

        BaseDateTime, LAT, LON, SOG, COG = ship_dict[mmsi][i+1]
        t1 = (pd.to_datetime(BaseDateTime) - start_time).seconds + (pd.to_datetime(BaseDateTime) - start_time).days *24*3600
        x1, y1 = geo_to_pixel(chart, [LON, LAT], extent)
        
        if t0 == t1:
            continue
        for j in range(t0, t1+1):
            interpolated_path[mmsi][j] = [x0+(x1-x0)*(j-t0)/(t1-t0), y0+(y1-y0)*(j-t0)/(t1-t0)]


interpolated_own_path = {}
final_own_path = own_path[-1]
final_arrival_time = [0]
for i in range(len(final_own_path)-1):
    x0, y0 = final_own_path[i]
    x1, y1 = final_own_path[i+1]
    dist = np.sqrt((x1-x0)**2 + (y1-y0)**2)
    delta_t = dist / own_speed
    final_arrival_time.append(final_arrival_time[-1] + delta_t)
for i in range(int(final_arrival_time[-1])):
    for j in range(len(final_arrival_time) - 1):
        if final_arrival_time[j] <= i and final_arrival_time[j+1] > i:
            break
    x0, y0 = final_own_path[j]
    x1, y1 = final_own_path[j+1]
    t0 = final_arrival_time[j]
    t1 = final_arrival_time[j+1]
    interpolated_own_path[i] = [x0+(x1-x0)*(i-t0)/(t1-t0), y0+(y1-y0)*(i-t0)/(t1-t0)]

with open(save_path + "log/" + "final_own_path.txt", 'a') as f:
    for x, y in final_own_path:
        f.write(str(x) + ", " + str(y) + "\n")

# In[24]:


# compare the length of path
human_path = []
for BaseDateTime, LAT, LON, SOG, COG in ship_dict[own_mmsi]:
    if start_time <= pd.to_datetime(BaseDateTime) <= end_time:
        human_path.append(geo_to_pixel(chart, [LON, LAT], extent))
        
human_dist = 0
for i in range(len(human_path) - 1):
    x0, y0 = human_path[i]
    x1, y1 = human_path[i+1]
    dist = np.sqrt((x1-x0)**2 + (y1-y0)**2)
    human_dist += dist

my_dist = 0
for i in range(len(final_own_path) - 1):
    x0, y0 = final_own_path[i]
    x1, y1 = final_own_path[i+1]
    dist = np.sqrt((x1-x0)**2 + (y1-y0)**2)
    my_dist += dist

# print("human path(yellow) length in pixels: ", human_dist)
# print("my path(green) length in pixels: ", my_dist)

with open(save_path + "metrics/" + "path_length.txt", 'a') as f:
    f.write("human path(yellow) length in pixels: " + str(human_dist) + "\n")
    f.write("my path(green) length in pixels: " +  str(my_dist) + "\n")

image = chart.copy()
draw_path(image, final_own_path, color=[0, 255, 0], width=10)
draw_path(image, human_path, color=[0, 255, 255], width=10)
cv2.imwrite(save_path + "metrics/" + "algorithm_vs_human.jpg", image)

# image = cv2.resize(image, [image.shape[1] // 10, image.shape[0] // 10])
# cv2.imshow("image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# In[25]:


# compare the p_local and max(p_local) cost
my_phi_list = []
for i in range(len(final_own_path) - 1):
    x0, y0 = final_own_path[i]
    x1, y1 = final_own_path[i+1]
    dist = np.sqrt((x1-x0)**2 + (y1-y0)**2)
    if dist == 0:
        continue
    phi = get_phi(x0, y0, x1, y1, dist)
    my_phi_list.append(phi)
    
my_p_local_list = []
for i in range(len(my_phi_list) - 1):
    phi0 = my_phi_list[i]
    phi1 = my_phi_list[i+1]
    p_local = abs(np.tan(0.5 * (phi0 - phi1)))
    my_p_local_list.append(p_local)

# print("my path, sum(p_local): ", sum(my_p_local_list), ", max(p_local): ", max(my_p_local_list))
with open(save_path + "metrics/" + "p_local.txt", 'a') as f:
    f.write("my path, sum(p_local): " + str(sum(my_p_local_list)) + ", max(p_local): " + str(max(my_p_local_list)) + "\n")

human_phi_list = []
for i in range(len(ship_dict[own_mmsi]) - 1):
    BaseDateTime, LAT, LON, SOG, COG = ship_dict[own_mmsi][i]
    t0 = (pd.to_datetime(BaseDateTime) - start_time).seconds + (pd.to_datetime(BaseDateTime) - start_time).days *24*3600
    x0, y0 = geo_to_pixel(chart, [LON, LAT], extent)
    if pd.to_datetime(BaseDateTime) < start_time:
        continue
        
    BaseDateTime, LAT, LON, SOG, COG = ship_dict[own_mmsi][i+1]
    t1 = (pd.to_datetime(BaseDateTime) - start_time).seconds + (pd.to_datetime(BaseDateTime) - start_time).days *24*3600
    x1, y1 = geo_to_pixel(chart, [LON, LAT], extent)
    if pd.to_datetime(BaseDateTime) > end_time:
        break
        
    dist = np.sqrt((x1-x0)**2 + (y1-y0)**2)
    if dist == 0:
        continue
    phi = get_phi(x0, y0, x1, y1, dist)
    human_phi_list.append(phi)

human_p_local_list = []
for i in range(len(human_phi_list) - 1):
    phi0 = human_phi_list[i]
    phi1 = human_phi_list[i+1]
    p_local = abs(np.tan(0.5 * (phi0 - phi1)))
    human_p_local_list.append(p_local)
# print("human path, sum(p_local): ", sum(human_p_local_list), ", max(p_local): ", max(human_p_local_list))
with open(save_path + "metrics/" + "p_local.txt", 'a') as f:
    f.write("human path, sum(p_local): " + str(sum(human_p_local_list)) + ", max(p_local): " + str(max(human_p_local_list)) + "\n")

# In[26]:


# compare the min dist to other ships
min_dist = 100000
human_dist_list = []
for i in interpolated_own_path:
    cur_min_dist = 100000
    if i not in interpolated_path[own_mmsi]:
        continue
    x0, y0 = interpolated_path[own_mmsi][i]
    for mmsi in interpolated_path:
        if mmsi != own_mmsi:
            if i in interpolated_path[mmsi]:
                x1, y1 = interpolated_path[mmsi][i]
                dist = np.sqrt((x1-x0)**2 + (y1-y0)**2)
                cur_min_dist = min(cur_min_dist, dist)
    min_dist = min(min_dist, cur_min_dist)
    human_dist_list.append(cur_min_dist)

# print("human path's min distance to other ships in pixels: ", min_dist)
with open(save_path + "metrics/" + "risk.txt", 'a') as f:
    f.write("human path's min distance to other ships in pixels: " + str(min_dist) + "\n")

min_dist = 100000
my_dist_list = []
for i in interpolated_own_path:
    x0, y0 = interpolated_own_path[i]
    cur_min_dist = 100000
    for mmsi in interpolated_path:
        if mmsi != own_mmsi:
            if i in interpolated_path[mmsi]:
                x1, y1 = interpolated_path[mmsi][i]
                dist = np.sqrt((x1-x0)**2 + (y1-y0)**2)
                cur_min_dist = min(cur_min_dist, dist)
    min_dist = min(min_dist, cur_min_dist)
    my_dist_list.append(cur_min_dist)

# print("my path's min distance to other ships in pixels: ", min_dist)
with open(save_path + "metrics/" + "risk.txt", 'a') as f:
    f.write("my path's min distance to other ships in pixels: " + str(min_dist) + "\n")
x0 = list(range(len(human_dist_list)))
x1 = list(range(len(my_dist_list)))
plt.plot(x0, human_dist_list, 'y')
plt.plot(x1, my_dist_list, 'g')
plt.legend(["human", "algorithm"])
plt.savefig(save_path + "metrics/" + "min_dist.png")
# plt.show()



# In[19]:


# path_list = possible_path[mmsi]["path_list"]
# arrival_time_list = possible_path[mmsi]["arrival_time_list"]
# weight_list = possible_path[mmsi]["weight_list"]
# for i in range(len(path_list)):
#     path = path_list[i]
#     arrival_time = arrival_time_list[i]
#     weight = weight_list[i]
#     risk_target = 0

#     x_target, y_target =  get_pos_prediction(path, arrival_time, pd.to_datetime(BaseDateTime), t)


# In[20]:


# image = chart.copy()
# draw_path(image, possible_path[mmsi]["path_list"][0])
# image = cv2.resize(image, [image.shape[1] // 10, image.shape[0] // 10])

# cv2.imshow("image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# In[21]:


# cv2.destroyAllWindows()

