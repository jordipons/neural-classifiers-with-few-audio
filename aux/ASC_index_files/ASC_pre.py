import pandas as pd
import numpy as np

metadata_train_path = "meta_dev.tsv" # where is the original metadata file?
metadata_test_path = "meta_eval.tsv" # where is the original metadata file?
name = "asc_fold" # name of the output file?
fold = 0
num_classes = 15
classes = ['beach', 'bus', 'cafe/restaurant', 'car', 'city_center', 'forest_path', 'grocery_store', 'home', 'library', 'metro_station', 'office', 'park', 'residential_area', 'train', 'tram']

train_items = pd.read_csv(metadata_train_path, delimiter='\t', header=None)
test_items = pd.read_csv(metadata_test_path, delimiter='\t', header=None)

index_w = open("index_"+name+"s_all.tsv", "w")
gt_tr_w = open("gt_"+name+str(fold)+"_train.tsv", "w")
gt_test_w = open("gt_"+name+str(fold)+"_test.tsv", "w")

for index_train, row in train_items.iterrows():
    index_w.write("%s\t%s\n" % (index_train,"TUT-acoustic-scenes-2017-development/"+row[0]))
    gt_vector = np.zeros(num_classes)
    gt_vector[classes.index(row[1])] = 1
    gt_tr_w.write("%s\t%s\n" % (index_train,gt_vector.tolist()))

print(row)
print(index_train)

for index_test, row in test_items.iterrows():
    index = index_train + 1 + index_test
    index_w.write("%s\t%s\n" % (index,"TUT-acoustic-scenes-2017-evaluation/"+row[0]))
    gt_vector = np.zeros(num_classes)
    gt_vector[classes.index(row[1])] = 1
    gt_test_w.write("%s\t%s\n" % (index,gt_vector.tolist()))

print(index_test)
print(index)

print('Done!')
