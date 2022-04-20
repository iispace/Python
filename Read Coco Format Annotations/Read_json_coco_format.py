def readCocoFormatJson(annot_file):
    """
    In Coco format, each bounding box is described using four values 
    [x_min, y_min, width, height] and category_id is described as an integer.
    Class mapping will be done from category_id to the class name.

    coco_root\JPEGImages\*.jpg
    coco_root\annotations.json
    """
    img_path = os.path.dirname(annot_file)

    all_class_labels = []
    coordinates = []

    with open(annot_file, 'r') as f:
        data = json.load(f)
    
    images_dic = []

    for image_dic in data['images']:
        i_dic = {} 
        i_dic['filename'] = image_dic['file_name']
        image_id = image_dic['id']
        i_dic['image_id'] = image_id
        category_ids = []
        bboxes = []
        for i, annot_dic in enumerate(data['annotations']):
            if (annot_dic['image_id'] == image_id):
                category_ids.append(annot_dic['category_id'])
                bboxes.append(annot_dic['bbox'])
        images_dic.append(i_dic)
        all_class_labels.append(category_ids)
        coordinates.append(bboxes)

    return all_class_labels, coordinates
