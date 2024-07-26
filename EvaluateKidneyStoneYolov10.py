# -*- coding: utf-8 -*-
"""
Created on Jul 2024

@author: Alfonso Blanco
"""
#######################################################################
# PARAMETERS
######################################################################
dir=""
dirname= "test1\\images"
dirnameLabels="test1\\labels"

import matplotlib.pyplot as plt
import matplotlib.patches as patches


#dirnameYolo="runs\\train\\exp2\\weights\\last.pt"

dirnameYolo="last39epoch.pt"

import cv2
import time
Ini=time.time()

#from ultralytics import YOLOv10 # after upgrade ultralytics
from ultralytics import YOLO


#model = YOLOv10(dirnameYolo)
model = YOLO(dirnameYolo)

class_list = model.model.names
print(class_list)

import numpy as np

import os
import re

import imutils

TotalStonesDetected=0
TotalStonesFalseDetected=0
TotalStones=0

########################################################################
def loadimages(dirname):
 #########################################################################
 # adapted from:
 #  https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
 # by Alfonso Blanco García
 ########################################################################  
     imgpath = dirname + "\\"
     
     images = []
     TabFileName=[]
   
    
     print("Reading imagenes from ",imgpath)
     NumImage=-2
     
     Cont=0
     for root, dirnames, filenames in os.walk(imgpath):
         print("LLEGAAAAAAAAAAAAAAAAAAAAA")
        
         NumImage=NumImage+1
         
         for filename in filenames:
             
             if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                 
                 
                 filepath = os.path.join(root, filename)
                
                 
                 image = cv2.imread(filepath)
                 #print(filepath)
                 #print(image.shape)                           
                 images.append(image)
                 TabFileName.append(filename)
                 
                 Cont+=1
     
     return images, TabFileName
########################################################################
def loadlabels(dirnameLabels):
 #########################################################################
 # adapted from:
 #  https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
 # by Alfonso Blanco García
 ########################################################################  
     imgpath = dirnameLabels + "\\"
        
     Labels = []
     TabFileLabelsName=[]
     Tabxyxy=[]
     
     ContLabels=0
     ContNoLabels=0
         
     print("Reading labels from ",imgpath)
        
     for root, dirnames, filenames in os.walk(imgpath):
         
         for filename in filenames:
                
                 TabLinxyxy=[]          
                 filepath = os.path.join(root, filename)
                
                 f=open(filepath,"r")

                 Label=""
                 xyxy=""
                 for linea in f:
                      linea = linea.replace('\n', ' ')
                      indexStone=int(linea[0])
                      Label=class_list[indexStone]
                      xyxy=linea[2:]
                      TabLinxyxy.append(xyxy)
                      
                                            
                 Labels.append(Label)
                 
                 if Label=="":
                      ContLabels+=1
                 else:
                     ContNoLabels+=1 
                 
                 TabFileLabelsName.append(filename)
                 Tabxyxy.append(TabLinxyxy)
     return Labels, TabFileLabelsName, Tabxyxy, ContLabels, ContNoLabels

# ttps://medium.chom/@chanon.krittapholchai/build-object-detection-gui-with-yolov8-and-pysimplegui-76d5f5464d6c
def DetectKidneyStoneWithYolov10 (img):
  
   TabcropKidneyStone=[]
   
   y=[]
   yMax=[]
   x=[]
   xMax=[]
   Tabclass_name=[]
   
   # https://blog.roboflow.com/yolov10-how-to-train/
   results = model(source=img)
  
   for i in range(len(results)):
       # may be several stones in a frame
       result=results[i]
       
       xyxy= result.boxes.xyxy.numpy()
       confidence= result.boxes.conf.numpy()
       class_id= result.boxes.cls.numpy().astype(int)
       #print(class_id)
       out_image = img.copy()
       for j in range(len(class_id)):
           con=confidence[j]
           label=class_list[class_id[j]] + " " + str(con)
           box=xyxy[j]
           
           cropKidneyStone=out_image[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
           
           TabcropKidneyStone.append(cropKidneyStone)
           y.append(int(box[1]))
           yMax.append(int(box[3]))
           x.append(int(box[0]))
           xMax.append(int(box[2]))

           # Tabclass_name only contains confidence, there is only a class and the name is not interesting
           Tabclass_name.append(label)
            
      
   return TabcropKidneyStone, y,yMax,x,xMax, Tabclass_name
def plot_image(image, box, boxesTrue, NameImage):
    
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    #Figure.suptitle(NameImage)
    fig.suptitle(NameImage)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    Cont=0
    

    for i in range(len(boxesTrue)):
         boxTrue=boxesTrue[i]
         upper_left_x_True = float(boxTrue[0]) - float( boxTrue[2] )/ 2.0
         upper_left_y_True = float(boxTrue[1]) - float( boxTrue[3]) / 2.0
         rect = patches.Rectangle(
                 (upper_left_x_True * width, upper_left_y_True * height),
                 float(boxTrue[2]) * width,
                 float(boxTrue[3]) * height,
                 linewidth=2,
                 edgecolor="green",
                 facecolor="none",
        )
        # Add the patch to the Axes
       
         ax.add_patch(rect)
   
    plt.show()

###########################################################
# MAIN
##########################################################

Labels, TabFileLabelsName, TabxyxyTrue, ContLabels, ContNoLabels= loadlabels(dirnameLabels)

imagesComplete, TabFileName=loadimages(dirname)

print("Number of images to test: " + str(len(imagesComplete)))

ContError=0
ContHit=0
ContNoDetected=0

for i in range (len(imagesComplete)):
 
            if TabFileLabelsName[i][:len(TabFileLabelsName[i])-4] != TabFileName[i][:len(TabFileName[i])-4]:
                 print("ERROR SEQUENCING IMAGES AN LABELS " + TabFileLabelsName[i][:len(TabFileLabelsName[i])-4] +" --" + TabFileName[i][:len(TabFileName[i])-4])
                 break
            # no se consideran las que no vienen labeladas
            if Labels[i] == "": continue
            gray=imagesComplete[i]
            
            imgTrue=imagesComplete[i]
            
                       
            TabImgSelect, y, yMax, x, xMax, Tabclass_name =DetectKidneyStoneWithYolov10(gray)
            #print(gray.shape)
            if TabImgSelect==[]:
                print(TabFileName[i] + " NON DETECTED")
                ContNoDetected=ContNoDetected+1 
                continue
            else:
                #ContDetected=ContDetected+1
                print(TabFileName[i] + " DETECTED ")
                
                
            for z in range(len(TabImgSelect)):
                #if TabImgSelect[z] == []: continue
                gray1=TabImgSelect[z]
                #cv2.waitKey(0)
                start_point=(x[z],y[z]) 
                end_point=(xMax[z], yMax[z])
                color=(255,0,0)
                # Using cv2.rectangle() method
                # Draw a rectangle with blue line borders of thickness of 2 px
                img = cv2.rectangle(gray, start_point, end_point,(255,0,0), 2)
                # Put text
                text_location = (x[z], y[z])
                text_color = (255,255,255)
                if Tabclass_name[z][:len(Labels[i])] !=Labels[i]:
                     #print(len(Tabclass_name[z]))
                     #print(len(Labels[i]))
                     print("ERROR " + TabFileName[i] + "Predicted "+ Tabclass_name[z] + " true is " + Labels[i])
                     ContError+=1
                else:
                     #print("HIT " + TabFileName[i] + "Predicted "+ Tabclass_name[z] )
                     ContHit+=1

                #  poner labels o conf enmarañan la imagen  
                #cv2.putText(img, str(Tabclass_name[z][len(Labels[i]):]) ,text_location
                #        , cv2.FONT_HERSHEY_SIMPLEX , 1
                #        , text_color, 2 ,cv2.LINE_AA)
                #cv2.putText(gray1, str(Tabclass_name[z][len(Labels[i]):]) ,text_location
                #        , cv2.FONT_HERSHEY_SIMPLEX , 1
                #        , text_color, 2 ,cv2.LINE_AA)
                        
                #cv2.imshow('Bone Fracture', gray1)
                #cv2.waitKey(0)
                #break
            #      
            #show_image=cv2.resize(img,(1000,700))
            #cv2.imshow('Frame', show_image)
            #cv2.imshow('Frame', img)
            #cv2.waitKey(0)
            boxes=[]
            
            TabLinxyxyTrue=TabxyxyTrue[i]
           
            boxesTrue=[]
            TabBoxesTrue=[]
            for z in range(len(TabLinxyxyTrue)):
                 
                 xyxyTrue=TabLinxyxyTrue[z].split(" ")
                
                 boxesTrue=[]
                 boxesTrue.append(float(xyxyTrue[0]))
                 boxesTrue.append(float(xyxyTrue[1]))
                 boxesTrue.append(float(xyxyTrue[2]))
                 boxesTrue.append(float(xyxyTrue[3]))
                 TabBoxesTrue.append(boxesTrue)
                 
               
            Stones=len(TabBoxesTrue)
            StonesDetected=len(TabImgSelect)
            
            print("there are " + str(Stones) + " stones.. " + " detected "+ str(StonesDetected))
            if StonesDetected> Stones:
                TotalStonesFalseDetected+=  StonesDetected-  Stones

            TotalStonesDetected+=StonesDetected
            TotalStones+=Stones
            
            plot_image(img, boxes, TabBoxesTrue, TabFileName[i])
           
             
              
print("")           
print("NO detected=" + str(ContNoDetected))
#print("Errors=" + str(ContError))
#print("Hits=" + str(ContHit))
print("")
print("Total Stones=" + str(TotalStones))
print("Total Stones Detected=" + str(TotalStonesDetected))
print("Total Stones False Detected=" + str(TotalStonesFalseDetected))

print( " Time in seconds "+ str(time.time()-Ini))
