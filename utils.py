from PIL import Image
import os
import numpy as np
import csv
import pandas as pd
from sklearn.utils import shuffle

def read_trainimgset(F,w):
    ### read training labels at level F and w####
    os.chdir("/home/jiaxin/PycharmProjects/SSC")#please change the directory to your working path
    training_image_name = []
    training_cell_count = []
    training_blur_level = []
    training_stain = []
    with open('train_label.csv', newline='') as f:
        reader = csv.reader(f)
        next(reader,None)
        for row in reader:
            training_image_name.append(row[0])
            training_cell_count.append(row[1])
            training_blur_level.append(row[2])
            training_stain.append(row[3])
    trainDF=pd.DataFrame(data={'blur':training_blur_level,'stain':training_stain,'count':training_cell_count},index=training_image_name)
    trainDF_F = trainDF.filter(regex=F, axis=0)
    trainDF_F_w = trainDF_F.filter(regex=w, axis=0)
    trainDF=shuffle(trainDF_F_w)

    ### read training images ####
    os.chdir("/home/jiaxin/Desktop/Data Files_Question1_SSC2019CaseStudy/train/")#please change the directory to your working path
    ##create a list (training_list) containing pixel value of 2400 training images
    training_list = []
    for img_name in trainDF_F_w.index:
        if (F and w) not in img_name:
            continue
        im = Image.open(img_name)
        imarray = np.array(im)
        imarray=imarray.reshape([520,696,1])
        training_list.append(imarray)
    return np.array(training_list), \
           np.array(trainDF_F_w['blur']).reshape([-1,1]),\
           np.array(trainDF_F_w['stain']).reshape([-1,1]),\
           np.array(trainDF_F_w['count']).reshape([-1,1])

def readtrainingset():
    os.chdir("/home/jiaxin/PycharmProjects/SSC")  # please change the directory to your working path
    training_image_name = []
    training_cell_count = []
    training_blur_level = []
    training_stain = []
    with open('train_label.csv', newline='') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            training_image_name.append(row[0])
            training_cell_count.append(row[1])
            training_blur_level.append(row[2])
            training_stain.append(row[3])

    ### read training images ####
    os.chdir(
        "/home/jiaxin/Desktop/Data Files_Question1_SSC2019CaseStudy/train/")  # please change the directory to your working path
    ##create a list (training_list) containing pixel value of 2400 training images
    training_list = []
    for i in training_image_name:
        im = Image.open(i)
        imarray = np.array(im)
        hist=np.histogram(imarray,range(256))[0]
        training_list.append(hist)
    training_img=np.array(training_list)
    trainingset=pd.DataFrame(training_img)
    trainingset['count'] = training_cell_count
    trainingset['blur'] = training_blur_level
    trainingset['stain'] = training_stain
    trainingset.index=training_image_name
    return trainingset

def get_dummy(set):
    return pd.get_dummies(set,columns=['blur','stain'],drop_first=True)

def get_traindata_level(F='F1',w='w1'):
    dataset=readtrainingset()
    datasetF1 = dataset.filter(regex=F, axis=0)
    datasetF1w1 = datasetF1.filter(regex=w, axis=0)
    return shuffle(datasetF1w1)



# ## read testing images ####
# os.chdir("/home/jiaxin/Desktop/Data Files_Question1_SSC2019CaseStudy/test/")#please change the directory to your working path
# ##create a list (testing_list) containing pixel value of 1200 testing images
# testing_list = []
# for i in testing_image_name:
# 	im = Image.open(i)
# 	#im.show()
# 	imarray = np.array(im)
# 	testing_list.append(imarray)



