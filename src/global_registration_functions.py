#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 09:07:39 2021

@author: sebastianhedegaardhansen
"""
import open3d as o3d
import numpy as np
import copy


def load_point_clouds(images, data_path, voxel_size=0.1):
    pcds = []
    
   # trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0], 
    #                         [0.0, 1.0, 0.0, 0.0],
     #                        [0.0, 0.0, 1.0, 0.0], 
      #                       [0.0, 0.0, 0.0, 1.0]])
    for i, img in enumerate(images):
        pcd = o3d.io.read_point_cloud(f"{data_path}/points{img}.ply")
        
        
        #pcd = o3d.io.read_point_cloud("TestData/cloud_bin_%d.pcd" % i)
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        #pcd_down.transform(trans_init)
    
        pcds.append(pcd_down)
    return pcds

def draw_teeth(list_of_teeth, transformation):
    list_of_teeth[0].transform(transformation)
    o3d.visualization.draw_geometries([list_of_teeth[0], list_of_teeth[1]])
    


def preprocess_point_cloud(pcd, voxel_size):
    
    #print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    #print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    #print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh    



def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.999))
    return result
    