import albumentations as A
import datetime
import numpy as np
import os, sys

label_file = r".\labels.txt"
all_transformed_class_labels = [[1], [1], [1], [1], [1], [1], [2], [2], [2], [2], [2], [3], [3], [3], [3], [3]]
all_transformed_bboxes = [[(0.0, 0.0, 256.0, 256.0)],  [(0.0, 205.19384152719664, 209.28136545566502, 50.80615847280336)],
                          [(0.0, 0.0, 256.0, 256.0)],  [(0.0, 0.0, 240.9343525654561, 256.0)],  [(35.671061934389115, 0.0, 220.32893806561088, 256.0)],
                          [(0.0, 0.0, 256.0, 207.04647163722825)], [(0.0, 0.0, 256.0, 246.08364932008365)], [(44.0, 77.0, 212.0, 179.0)],
                          [(0.0, 0.0, 256.0, 198.0)], [(0.0, 0.0, 256.0, 142.0)], [(55.0, 0.0, 201.0, 174.0)],
                          [(0.0, 0.0, 256.0, 256.0)], [(0.0, 0.0, 256.0, 157.0)], [(0.0, 2.552375562263258, 210.54226670506912, 253.44762443773675)],
                          [(0.0, 0.0, 256.0, 256.0)], [(0.0, 0.0, 227.67548999451753, 152.78459063914028)]]
#all_transformed_bboxes = [[[110, 122, 311, 166]],
#                         [[110, 122, 311, 166]],[[110, 122, 311, 166]],[[110, 122, 311, 166]],
#                         [[110, 122, 311, 166]],[[110, 122, 311, 166]],[[110, 122, 311, 166]],
#                         [[110, 122, 311, 166]],[[110, 122, 311, 166]],[[110, 122, 311, 166]],
#                         [[110, 122, 311, 166]],[[110, 122, 311, 166]],[[110, 122, 311, 166]],
#                         [[110, 122, 311, 166]],[[110, 122, 311, 166]],[[110, 122, 311, 166]]]

now = datetime.datetime.now()
now_str = now.strftime("%Y-%m-%d %H:%M:%S")
        
data = dict(
        info=dict(
        url=None,
        version=None,
        year=now.year,
        contributor=None,
        date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[dict(url=None, id=0, name=None,)],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type="instances",
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )
    
  
for i, (t_img, bboxes, class_labels, area) in enumerate(zip(all_transformed_images, all_transformed_bboxes, all_transformed_class_labels, bbox_areass)):
    height = t_img.shape[0]
    width = t_img.shape[1]
    annot_dic = dict(id=i, image_id=i, category_id= class_labels, segmentation=None, area=area, bbox=bboxes, iscrowd=0)
    data['annotations'].append(annot_dic)
    img_dic = dict(license=None, url=None, id=i, file_name=f"JPEGImages\\T{i}.jpg", height=height, width=width, date_captured=now_str)
    data['images'].append(img_dic)


class_name_to_id = {}
with open(label_file, 'r') as f:
    for i, line in enumerate(f.readlines()):
        class_id = i - 1
        class_name = line.strip()
        if class_id == -1:
            assert class_name == '__ignore__'
            continue
        class_name_to_id[class_name] = class_id
        data['categories'].append(dict(supercategory=None, id=class_id, name=class_name))

dst_folder = r"D:\Image_Seg\image_sources\BboxAnnoted_coco"
out_ann_file = os.path.join(dst_folder, "annotations.json")

with open(out_ann_file, 'w') as f:
    json.dump(data, f)
