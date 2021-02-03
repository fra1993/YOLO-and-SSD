import numpy as np
import tensorflow as tf
import pandas as pd
import os
import cv2 as cv
import json
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from math import ceil

class BatchGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, batch_size, coco, grid_size, images_dim, shuffle, directory, num_classes):
        self.batch_size = batch_size
        self.coco = coco
        self.catIds = coco.getCatIds(catNms=coco.cats)
        self.imgIds = coco.getImgIds(catIds=self.catIds)
        self.grid_size = grid_size 
        self.img_dim = images_dim
        self.shuffle = shuffle
        self.dir_ = directory
        self.num_classes = num_classes
        self.version = 2.0
        self.on_epoch_end()
        
    def on_epoch_end(self):
      'Updates indexes after each epoch'
      self.indexes = np.arange(len(self.imgIds))
      if self.shuffle == True:
          np.random.shuffle(self.indexes)
            
    def Parse_annotations(self,annotations,image_H,image_W):
        '''
        This function returns a list of lists. Each element of the main list contains the parameters of a of a bounding box:

        - x: center coordinate x 
        - y: center coordinate y 
        - w: width of the bounding box
        - h: height of the bounding box
        - class: class of the object inside the bounding box

        '''
        data = []
        for annotation in annotations:
            bbox = self.Rescale_BB_coordinates(annotation['bbox'],image_H,image_W)
            category_id = annotation['category_id']
            bbox.append(category_id)
            data.append(bbox)
        return data

    def Rescale_BB_coordinates(self,data,image_H,image_W):
        '''rescale BB coordinates according to iamge dimensions:
        - output: center_x, center_y, width, height 
        - input: upper_left_x, upper_left_y, width, height'''
        n_data = []
        n_data.append((data[0]+data[2]/2)/image_W) # x
        n_data.append((data[1]+data[3]/2)/image_H) # y
        n_data.append((data[2])/image_W) # w 
        n_data.append((data[3])/image_H) # h
        return n_data
 
    def Build_detection_grid(self,annotations,grid_size,num_classes,priors=[(1.0,1.0)]):
        '''This function builds the detection grid associated with each image:
        - input: image_size = (image_Width, image_Height)
        - output: (grid_size[0],grid_size[1],B*(5+C)) where B is the number of priors and C the number of classes 
        '''

        # initialize the grid
        grid_volume = (self.grid_size[0],self.grid_size[1],len(priors)*(5+self.num_classes))
        grid = np.zeros(grid_volume)

        for annotation in annotations:

            # unpack annotations
            x,y,w,h,label = annotation

            # find the Grid Cell (GC) responsible for the detection        
            resp_GC_x = int(x*self.grid_size[0])
            resp_GC_y = int(y*self.grid_size[1])

            for prior in priors:

                # create labels vector
                labels = np.zeros(self.num_classes)
                # pay attention that the correct label is equal to the position where labels==1 +1
                labels[label-1]=1

                # fill responsible Grid Cell (1:boxiness)
                grid[resp_GC_x,resp_GC_y,:5] = x,y,w,h,1
                grid[resp_GC_x,resp_GC_y,5:] = labels

        return grid

    def __data_generation(self, list_Ids_temp):
        # iamges batch
        batch = []
        batch_anns  = []
        detection_grid = []

        for i, Id in enumerate(list_Ids_temp):
            
            # random select an image and its annotations
            img = self.coco.loadImgs(Id)[0]
            annIds = self.coco.getAnnIds(imgIds=img['id'], catIds=self.catIds, iscrowd=None)

            ### load the image and the annotations ###
            image = cv.imread(os.path.join(self.dir_,img["file_name"]))

            # cvtColor changes the color conversion
            rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image_H = rgb_image.shape[0]
            image_W = rgb_image.shape[1]

            # resize the image
            rgb_image = cv.resize(rgb_image, self.img_dim)

            #load annotations
            anns = self.coco.loadAnns(annIds)

            # append the image to the batch and annotations to batch_anns
            batch.append(rgb_image)
            temp_annotations = self.Parse_annotations(anns,image_H,image_W)
            batch_anns.append(temp_annotations)       

            ## build detection grid
            detection_grid.append(self.Build_detection_grid(temp_annotations,self.grid_size,self.num_classes))

        return (np.array(batch, dtype='object').astype('float32'), np.array(detection_grid, dtype='object').astype('float32'))
    
    
    def __len__(self):
      'Denotes the number of batches per epoch'
      return int(np.floor(len(self.imgIds) / self.batch_size))

    def __getitem__(self, index):
        
      'Generate one batch of data'
      # Generate indexes of the batch
      indexes = np.random.randint(0,len(self.imgIds),self.batch_size)

      # Find list of IDs
      list_IDs_temp = [self.imgIds[k] for k in indexes]

      # Generate data
      X, y = self.__data_generation(list_IDs_temp)

      return X, y
