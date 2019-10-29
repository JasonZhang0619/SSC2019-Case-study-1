from PIL import Image
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def read_imgset(csv_path='train_label.csv',train=True, F=None, w=None, hist = True):
    ### read training labels at level F and w
    df=pd.read_csv(csv_path,index_col='image_name')
    if (F):
        df = df.filter(regex=F, axis=0)
    if (w):
        df = df.filter(regex=w, axis=0)
    if(train):
        dir = "train/"
    else:
        dir= "test/"
    ##create a list (training_list) containing pixel value ofimages
    img_list = []
    for img_name in df.index:
        if (F and w) not in img_name:
            continue
        im = Image.open(dir+img_name)
        imarray = np.array(im)
        if(hist):
            imarray = np.histogram(imarray, range(256))[0] #read hist
        else:
            imarray=imarray.reshape([520,696,1]) # read original img
        img_list.append(imarray)
    return np.array(img_list), \
           df

def get_dummy(set):
    return pd.get_dummies(set,columns=['blur','stain'],drop_first=True)






