#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 09:02:30 2021

@author: sebastianhedegaardhansen
"""
import open3d as o3d
import numpy as np
import copy
import global_registration_functions as grf



images = ['0', '01', '02']#  '19']'07', '08', '09']
data_path ="/Users/sebastianhedegaardhansen/Documents/DTU/9.semester/Project_in_image_analysis/02507-Project_work_Image_Analysis_and_Computer_Graphics/data/plyfolder"

#load images
teeth = grf.load_point_clouds(images, data_path)

################ preprocess data ################
voxel_size = 0.1  # means 5cm for this dataset

teeth_prepared = []
teeth_fpfh = []
for pcd in teeth:
    pcd_down, pcd_fpfh = grf.preprocess_point_cloud(pcd, voxel_size)
    teeth_prepared.append(pcd_down)
    teeth_fpfh.append(pcd_fpfh)



teeth_prepared[0], outlier_index = teeth_prepared[0].remove_radius_outlier(
                                              nb_points=21,
                                              radius=0.5)

teeth_prepared[1], outlier_index = teeth_prepared[1].remove_radius_outlier(
                                              nb_points=21,
                                              radius=0.5)




################ Loop to stitch all the pointClouds #################
collectedPointCLoud = teeth_prepared[0]
fpfh_collected = teeth_fpfh[0]

for i in range(1, len(teeth_prepared)):
    point_cloud_global = grf.execute_global_registration(teeth_prepared[i], collectedPointCLoud,
                                                    teeth_fpfh[i], fpfh_collected, voxel_size)
    teeth_prepared[i] = teeth_prepared[i].transform(point_cloud_global.transformation)
    grf.draw_teeth([teeth_prepared[i], collectedPointCLoud], point_cloud_global.transformation)
    
    threshold = 0.5
    #trans_init  = point_cloud_global.transformation
    
    trans_init = np.asarray([[1,0,0,0],   # 4x4 identity matrix, this is a transformation matrix,
                             [0,1,0,0],   # It means there is no displacement, no rotation, we enter
                             [0,0,1,0],   # This matrix is ​​the initial transformation
                             [0,0,0,1]])
    
    point_cloud = o3d.pipelines.registration.registration_icp(
        teeth_prepared[i], collectedPointCLoud, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    
    grf.draw_teeth([teeth_prepared[i], collectedPointCLoud], point_cloud.transformation)
    print(point_cloud)
    collectedPointCLoud = teeth_prepared[i].transform(point_cloud.transformation) + collectedPointCLoud
    collectedPointCLoud.paint_uniform_color([1, 0.706, 0])
    fpfh_collected = o3d.pipelines.registration.compute_fpfh_feature(collectedPointCLoud,
                                                                               o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    
o3d.visualization.draw_geometries([collectedPointCLoud])


#combine 

#collectedPointCLoud.paint_uniform_color([1, 0.706, 0])
#o3d.visualization.draw_geometries([collectedPointCLoud])
