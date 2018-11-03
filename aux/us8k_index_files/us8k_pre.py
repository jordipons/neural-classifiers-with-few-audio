import pandas as pd
import numpy as np

metadata_path = "UrbanSound8K.csv" # where is the original metadata file?
name = "us8k_fold" # name of the output file?
fold = 2
test_folder = fold + 1 # except for fold=9  then 10
val_folder = fold + 2 # except for fold=9 then 1
num_classes = 10

df_items = pd.read_csv(metadata_path)

index_w = open("index_"+name+"s_all.tsv", "w")
gt_tr_w = open("gt_"+name+str(fold)+"_train.tsv", "w")
gt_val_w = open("gt_"+name+str(fold)+"_val.tsv", "w")
gt_test_w = open("gt_"+name+str(fold)+"_test.tsv", "w")

for index, row in df_items.iterrows():
    index_w.write("%s\t%s\n" % (index,"fold"+str(row.fold)+"/"+row.slice_file_name))
    gt_vector = np.zeros(num_classes)
    gt_vector[row.classID-1] = 1
    if row.fold == test_folder: # test set
        gt_test_w.write("%s\t%s\n" % (index,gt_vector.tolist()))
    elif row.fold == val_folder: # validation set
        gt_val_w.write("%s\t%s\n" % (index,gt_vector.tolist()))
    else: # training set
        gt_tr_w.write("%s\t%s\n" % (index,gt_vector.tolist()))

print('Done!')
