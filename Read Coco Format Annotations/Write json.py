label_file = r".\labels.txt"
all_class_labels = [[1],[1],[1],[1],[1],[1],[2],[2],[2],[2],[3],[3],[3],[3],[3],[3]]
all_transformed_bboxes = [[[110, 122, 311, 166]],
                         [[110, 122, 311, 166]],[[110, 122, 311, 166]],[[110, 122, 311, 166]],
                         [[110, 122, 311, 166]],[[110, 122, 311, 166]],[[110, 122, 311, 166]],
                         [[110, 122, 311, 166]],[[110, 122, 311, 166]],[[110, 122, 311, 166]],
                         [[110, 122, 311, 166]],[[110, 122, 311, 166]],[[110, 122, 311, 166]],
                         [[110, 122, 311, 166]],[[110, 122, 311, 166]],[[110, 122, 311, 166]]]

now = datetime.datetime.now()
now_str = now.strftime("%Y-%m-%d")
        
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
    
for i, (bboxes, class_labels) in enumerate(zip(all_transformed_bboxes, all_class_labels)):
    annot_dic = dict(segmentation=None, area=None, iscrowd=0, image_id=i, bbox=bboxes, category_id= class_labels)
    data['annotations'].append(annot_dic)
    img_dic = dict(license=None, url=None, file_name=f"{i}.jpg", height=height, width=width, date_captured=now_str, id=i)
    data['images'].append(img_dic)
      

for i, (bboxes, class_labels) in enumerate(zip(all_transformed_bboxes, all_class_labels)):
    dic = dict(segmentation=None, area=None, iscrowd=0, image_id=i, bbox=bboxes, category_id= class_labels)
    data['annotations'].append(dic)
      
with open(label_file, 'r') as f:
    for i, line in enumerate(f.readlines()):
        class_id = i - 1
        class_name = line.strip()
        if class_id == -1:
            assert class_name == '__ignore__'
            continue
        class_name_to_id[class_name] = class_id
        data['categories'].append(dict(supercategory=None, id=class_id, name=class_name))


