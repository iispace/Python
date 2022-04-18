import json

root = r'D:\JsonPath'
json_file = 'instance_annotations.json'

with open(os.path.join(root, json_file), 'r') as f:
  data = json.load(f)
  
print(json.dumps(data, indent=4))

"""
[
    {
        "image": "000000386298.jpg",
        "annotations": [
            {
                "label": "cat",
                "coordinates": {
                    "x": 97.60152284263958,
                    "y": 284.89086294416245,
                    "width": 195.0,
                    "height": 217.0
                }
            },
            {
                "label": "dog",
                "coordinates": {
                    "x": 554.0507614213197,
                    "y": 219.89086294416245,
                    "width": 171.8984771573604,
                    "height": 247.0
                }
            }
        ]
    }
]
"""

images = []
classess = []
coordinatess = []

for datum in data:
    images.append(datum['image'])
    classes = []
    coordinates = []
    for annotation in datum['annotations']:
        classes.append(annotation['label'])
        coordinates.append(annotation['coordinates'])
    classess.append(classes)
    coordinatess.append(coordinates)
    
    
for i, (image, classes, coordinates) in enumerate(zip(images, classess, coordinatess)):
    print(f'image[{i}]: {image}')
    for j, (_class, _coordinates) in enumerate(zip(classes, coordinates)):
        print(f'class[{i}][{j}]:       {_class}')
        print(f'coordinates[{i}][{j}]: {_coordinates}')

"""
image[0]: 000000386298.jpg
class[0][0]:       cat
coordinates[0][0]: {'x': 97.60152284263958, 'y': 284.89086294416245, 'width': 195.0, 'height': 217.0}
class[0][1]:       dog
coordinates[0][1]: {'x': 554.0507614213197, 'y': 219.89086294416245, 'width': 171.8984771573604, 'height': 247.0}
"""
