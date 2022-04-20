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
    
before = data['categories']
print('Before appending any data:')
print(json.dumps(before, indent=4)
print("\n")

with open(label_file, 'r') as f:
    for i, line in enumerate(f.readlines()):
        class_id = i - 1
        class_name = line.strip()
        #print(f'LINE [{i}] : ', class_name)
        if class_id == -1:
            assert class_name == '__ignore__'
            continue
        #print(f'LINE [{i}] : ', class_name)
        class_name_to_id[class_name] = class_id
        data['categories'].append(dict(supercategory=None, id=class_id, name=class_name))
print("After appending data:")
print(json.dumps(data['categories'], indent=4))
