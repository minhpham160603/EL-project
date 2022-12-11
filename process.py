import xml.etree.ElementTree as ET
import os
import glob
import cv2
def xml_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = round(((bbox[2] + bbox[0]) / 2) / w, 5)
    y_center = round(((bbox[3] + bbox[1]) / 2) / h, 5)
    width = round((bbox[2] - bbox[0]) / w, 5)
    height = round((bbox[3] - bbox[1]) / h, 5)
    return [x_center, y_center, width, height]

input_dir = "./Synthesis_set/Synthesis-batch1"
files = glob.glob(os.path.join(input_dir, '*.xml'))

def process_file(file):
    basename = os.path.basename(file)
    filename = os.path.splitext(basename)[0]
    print(filename)
    xml_file = os.path.join(input_dir, filename + ".xml")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    element = root.find("size")
    h, w = int(element.find("height").text), int(element.find("width").text)

    objects = root.findall("object")

    [obj1, obj2, obj3], nh, nw = get_bbox(objects)
    bh, bw = obj1[2], obj1[3]
    horizontal = abs(obj3[0] - obj1[0])//nw
    vertical = abs(obj2[1] - obj1[1])//nh
    x_start = obj1[0]
    y_start = obj1[1]
    print(filename)
    print(obj1, obj2, obj3)
    res = []
    img = cv2.imread(os.path.join(input_dir, f"{filename}.jpg"))
    for i, center_y in enumerate(range(y_start, h, vertical)):
        for j, center_x in enumerate(range(x_start, w, horizontal)):
            xmin, ymin, xmax, ymax = get_yolo_box([center_x, center_y, bh, bw])
            txt_bbox = [str(x) for x in xml_to_yolo_bbox([xmin, ymin, xmax, ymax], w, h)]
            s = "0 " + " ".join(txt_bbox) 
            res.append(s)
            cv2.imwrite(f"./Cells1/{filename}-Cells{i}_{j}.jpg", img[ymin:ymax, xmin:xmax, :])

    with open(f"./Synthesis_set/Batch-1-txt/{filename}.txt", 'w') as outfile:
        outfile.write("\n".join(res))

def get_bbox(objects):
    object_pos = []
    nh, nw = 0, 0
    for obj in objects:
        position = obj.find("bndbox")
        label = obj.find("name").text
        if "," in label:
            tmp = label.split(",")
            nh, nw = int(tmp[0]), int(tmp[1])
        xmin, ymin, xmax, ymax = int(position.find("xmin").text), int(position.find("ymin").text), int(position.find("xmax").text), int(position.find("ymax").text)
        tmp = get_center([xmin, ymin, xmax, ymax]) #center x, center y, bh, bw
        object_pos.append(tmp)
    object_pos.sort(key=lambda item: item[0] + item[1])
    return object_pos, nh, nw

def get_center(bbox):
    center_x = (bbox[2] + bbox[0])//2
    center_y = (bbox[3] + bbox[1])//2
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    return center_x, center_y, bh, bw

def get_yolo_box(bbox):
    center_x, center_y, bh, bw = bbox
    xmin = center_x - bw//2
    xmax = center_x + bw//2
    ymin = center_y - bh//2
    ymax = center_y + bh//2
    return xmin, ymin, xmax, ymax

for file in files:
    process_file(file)
