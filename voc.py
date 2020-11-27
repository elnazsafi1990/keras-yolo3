import numpy as np
import os
import xml.etree.ElementTree as ET
import pickle
import csv
import cv2

def parse_voc_annotation(ann_dir, img_dir, cache_name, labels=[]):
    if os.path.exists(cache_name):
        with open(cache_name, 'rb') as handle:
            cache = pickle.load(handle)
        all_insts, seen_labels = cache['all_insts'], cache['seen_labels']
    else:
        all_insts = []
        seen_labels = {}
        names={}
        with open(ann_dir, newline='',mode='r') as csvfile:
          csvreader=csv.reader(csvfile)
          for row in csvreader:
            if row[0] not in names:
              names[row[0]]=[]
            names[row[0]].append(row)

        for name in names:
            img = {'object':[]}
            img['filename']=os.path.join(img_dir,name)
            img['width']=cv2.imread(os.path.join(img_dir,name)).shape[1] 
            img['height']=cv2.imread(os.path.join(img_dir,name)).shape[0] 
            for item in names[name]:
              img['object'] += [{'name':item[5],'xmin':item[1],'ymin':item[2],'xmax':item[3],'ymax':item[4]}]
              if item[5] in seen_labels:
                seen_labels[item[5]] += 1
              else:
                seen_labels[item[5]] = 1

            if len(img['object']) > 0:
                all_insts += [img]

        cache = {'all_insts': all_insts, 'seen_labels': seen_labels}
        with open(cache_name, 'wb') as handle:
            pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)    
                        
    return all_insts, seen_labels
