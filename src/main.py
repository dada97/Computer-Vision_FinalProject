import os

from numpy.core.fromnumeric import shape
import cv2
import numpy as np

image_dir = "./SidebySide/"
Calibration_data = image_dir+"CalibrationData.txt"
output_dir = "./Output/"
result = open(output_dir+"result.xyz", "a")

#find img feature point 
def find_feature(img):
    height, width, channel = img.shape
    features = []
    for y in range(height):
        xIndex = []
        color_values = []
        
        #find brightest pixel in each row
        for x in range(width):     
            pixel = img[y][x]
            color_value  = float(pixel[0])+float(pixel[1])+float(pixel[2])         
            if(color_value>100):
                xIndex.append(x)
                color_values.append(color_value)               
        if len(xIndex)>3:
            max_index = (xIndex[color_values.index(max(color_values))])
            features.append([y,max_index,1.0])       
    return np.array(features)  

#calculate 3D by direct triangulation
def directTriangulation(L_K,L_RT,R_K,R_RT,uv0,uv1):
    L = L_K@L_RT;
    R = R_K@R_RT;
    A = np.array([
        uv0[0]* L[2]-L[0],
        uv0[1]* L[2]-L[1],
        uv1[0]* R[2]-R[0],
        uv1[1]* R[2]-R[1]
    ])
    U,S,V = np.linalg.svd(np.array(A))
    X = V[-1]
    X = X/X[-1]
    return X


# read Calibration data
f = open(Calibration_data, "r")
lines = f.readlines()

left_cam_K = np.empty((0,3))
left_cam_RT = np.empty((0,4))
right_cam_K = np.empty((0,3))
right_cam_RT = np.empty((0,4))
fun_mat = np.empty((0,3))

for num, line in enumerate(lines, 1):
    if "#Left Camera K (3x3)" in line:
        for i in range(3):
            arr = [float(n) for n in lines[num+i].split()]
            left_cam_K = np.append(left_cam_K,[arr],axis=0)
    
    if "#Left Camera RT (3x4)" in line:
        for i in range(3):
            arr = [float(n) for n in lines[num+i].split()]
            left_cam_RT = np.append(left_cam_RT,[arr],axis=0)

    if "#Right Camera K (3x3)" in line:
        for i in range(3):
            arr = [float(n) for n in lines[num+i].split()]
            right_cam_K = np.append(right_cam_K,[arr],axis=0)
    
    if "#Right Camera RT (3x4)" in line:
        for i in range(3):
            arr = [float(n) for n in lines[num+i].split()]
            right_cam_RT = np.append(right_cam_RT,[arr],axis=0)

    if "#Fundamental Matrix (3x3)" in line:
        for i in range(3):
            arr = [float(n) for n in lines[num+i].split()]
            fun_mat = np.append(fun_mat,[arr],axis=0)

# processsing all images
for index in range(193):
    img_name ="SBS_"+str(index).zfill(3)+".jpg"
    print(img_name)
        
    img = cv2.imread(image_dir+img_name)

    img_L = img[0:img.shape[0],0:int(img.shape[1]/2)]
    img_R = img[0:img.shape[0],int(img.shape[1]/2):img.shape[1]]

    #find image feature
    feature_L = find_feature(img_L)
    feature_R = find_feature(img_R)

    for featurePoint_L in feature_L:
        minErrPoint = None
        minErr = img.shape[0]
        #match feature by compute fundanmental error
        for featurePoint_R in feature_R:
            a = np.array([featurePoint_L[1],featurePoint_L[0],featurePoint_L[2]])
            b = np.array([featurePoint_R[1],featurePoint_R[0],featurePoint_R[2]])
            error = abs(a.T@fun_mat.T@b)
            if(error<minErr):
                minErrPoint = featurePoint_R 
                minErr = error

        #if find the corresponding point, then use direct Triangulation Method to find 3d point
        if minErrPoint is not None:
            point = directTriangulation(left_cam_K,left_cam_RT,right_cam_K,right_cam_RT,np.array([featurePoint_L[1],featurePoint_L[0],featurePoint_L[2]]),np.array([minErrPoint[1],minErrPoint[0],minErrPoint[2]]))
            #write result ".xyz"
            result.writelines("{0} {1} {2}\n".format(point[0],point[1],point[2]))


result.close()

    
