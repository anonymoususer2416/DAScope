# ----------------------------------------------
# Created by Wei-Jie Huang
# A collection of annotation conversion scripts
# We provide this for reference only
# ----------------------------------------------


import json
from pathlib import Path
from tqdm import tqdm
import os

from PIL import Image
import xml.etree.ElementTree as ET
from box_ops import box_xyxy_to_cxcywh



r"""
coco_anno_dict is like {
    "images": list of image_info's
    "annotations": list of annotation_info's
    "categories": list of categorie_info's
}
where img_info is like: {
    "id": ...,                      # 0-indexed
    "width": ...,
    "height": ...,
    "file_name": ...,
}, annotation_info is like: {
    "id": ...,                      # 0-indexed
    "image_id": ...,
    "category_id": ...,
    "segmentation": ...,
    "iscrowd": ...,
    "area": ...,
    "bbox": ...,                    # (x, y, w, h)
}, and category_info is like: {
    "id": ...,                      # 1-indexed
    "name": ...,
}
"""


def sim10k_to_coco(
    src_path: str = "VOC2012/Annotations",
    des_path: str = "annotations/sim10k_caronly.json",
    categories: tuple = ("car",)
    ) -> None:

    r""" Convert Sim10k (in VOC format) into COCO format.
    Args:
        src_path: path of the directory containing VOC-format annotations
        des_path: destination of the converted COCO-fomat annotation
        categories: only category ``car`` is considered by default
    """

    from xml.etree import ElementTree

    src_path = Path(src_path)
    des_path = Path(des_path)
    assert src_path.exists(), "Annotation directory does not exist"
    if des_path.exists():
        print(f"{des_path} exists. Override? (y/n)", end=" ")
        if input() != "y":
            print("Abort")
            return
    else:
        des_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialization
    coco_anno_dict = {
        "images": [],
        "categories": [],
        "annotations": [],
    }
    num_images = 0
    num_categories = 0
    num_annotations = 0

    # Categories
    category_to_id = {}
    for category in categories:
        coco_anno_dict["categories"].append({
            "id": num_categories + 1,
            "name": category
        })
        category_to_id[category] = num_categories + 1
        num_categories += 1

    # Start Conversion
    for anno_file in tqdm(list(src_path.glob("*.xml"))):
        et_root = ElementTree.parse(anno_file).getroot()

        ##### Images #####
        img_info = {
            "id": num_images,
            "file_name": anno_file.stem + ".jpg",
        }
        num_images += 1

        # Image Size
        size = et_root.find("size")
        img_info["width"] = int(size.find("width").text)
        img_info["height"] = int(size.find("height").text)

        coco_anno_dict["images"].append(img_info)

        ##### Annotations #####
        for anno_object in et_root.findall("object"):
            category = anno_object.find("name").text
            if category not in categories:
                continue
            anno_info = {
                "id": num_annotations,
                "image_id": img_info["id"],
                "category_id": category_to_id[category],
                "segmentation": [],
                "iscrowd": 0
            }
            num_annotations += 1

            # Bounding box
            bndbox = anno_object.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            # COCO format expects (x, y, w, h)
            anno_info["bbox"] = [xmin, ymin, round(xmax - xmin, 2), round(ymax - ymin, 2)]
            anno_info["area"] = round(anno_info["bbox"][2] * anno_info["bbox"][3], 2)

            coco_anno_dict["annotations"].append(anno_info)

    print("# of images:", num_images)
    print("# of categories:", num_categories)
    print("# of annotations:", num_annotations)

    with open(des_path, 'w') as f:
        f.write(json.dumps(coco_anno_dict, indent=4))
    print(f"Convert successfully to {des_path}")


def bdd100k_daytime_to_coco(
    src_path: str = "labels/bdd100k_labels_images_train.json",
    des_path: str = "annotations/bdd_daytime_train.json",
    categories: tuple = (
        "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle")
    ) -> None:

    r""" Extract ``daytime`` subset from BDD100k dataset and convert into COCO format.
    Args:
        src_path: source of the annotation json file
        des_path: destination of the converted COCO-fomat annotation
        categories: categories used
    """

    src_path = Path(src_path)
    des_path = Path(des_path)
    assert src_path.exists(), "Source annotation file does not exist"
    if des_path.exists():
        print(f"{des_path} exists. Override? (y/n)", end=" ")
        if input() != "y":
            print("Abort")
            return
    else:
        des_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialization
    coco_anno_dict = {
        "images": [],
        "categories": [],
        "annotations": [],
    }
    num_images = 0
    num_categories = 0
    num_annotations = 0

    # Categories
    category_to_id = {}
    for category in categories:
        coco_anno_dict["categories"].append({
            "id": num_categories + 1,
            "name": category
        })
        category_to_id[category] = num_categories + 1
        num_categories += 1

    with open(src_path, 'r') as f:
        raw_img_annos = json.load(f)
    # Start Conversion
    for raw_img_anno in tqdm(raw_img_annos):
        if raw_img_anno["attributes"]["timeofday"] != "daytime":
            continue

        ##### Images #####
        img_info = {
            "id": num_images,
            "file_name": raw_img_anno["name"],
            "height": 720,
            "width": 1280
        }
        coco_anno_dict["images"].append(img_info)
        num_images += 1

        ##### Annotations #####
        for label in raw_img_anno["labels"]:
            if label["category"] not in category_to_id or "box2d" not in label:
                continue
            anno_info = {
                "id": num_annotations,
                "image_id": img_info["id"],
                "category_id": category_to_id[label["category"]],
                "segmentation": [],
                "iscrowd": 0,
            }
            num_annotations += 1

            # Bbox
            x1 = label["box2d"]["x1"]
            y1 = label["box2d"]["y1"]
            x2 = label["box2d"]["x2"]
            y2 = label["box2d"]["y2"]
            anno_info["bbox"] = [x1, y1, x2 - x1, y2 - y1]
            anno_info["area"] = float((x2 - x1) * (y2 - y1))
            coco_anno_dict["annotations"].append(anno_info)

    print("# of images:", num_images)
    print("# of categories:", num_categories)
    print("# of annotations:", num_annotations)

    with open(des_path, 'w') as f:
        f.write(json.dumps(coco_anno_dict, indent=4))
    print(f"Convert successfully to {des_path}")


def cityscapes_to_coco(
    src_path: str = "gtFine/train",
    des_path: str = "annotations/cityscapes_train.json",
    car_only: bool = False,
    foggy: bool = False,
    categories: tuple = (
        "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle")
    ) -> None:

    r"""Convert Cityscapes into COCO format.
        Ref: https://github.com/facebookresearch/Detectron/blob/7aa91aa/tools/convert_cityscapes_to_coco.py
    Args:
        src_path: path of the directory containing Cityscapes annotations
        des_path: destination of the converted COCO-fomat annotation
        car_only: whether extract category ``car`` only. used in Syn-to-real adaptation
        foggy: whether extract from foggy cityscapes. used in weather adaptation
        categories: categories used
    """

    def get_instances_with_polygons(imageFileName):
        r""" Ref: https://github.com/facebookresearch/Detectron/issues/111#issuecomment-363425465"""
        import os
        import sys
        import cv2
        import numpy as np
        from PIL import Image
        from cityscapesscripts.evaluation.instance import Instance
        from cityscapesscripts.helpers.csHelpers import labels, id2label

        # Load image
        img = Image.open(imageFileName)

        # Image as numpy array
        imgNp = np.array(img)

        # Initialize label categories
        instances = {}
        for label in labels:
            instances[label.name] = []

        # Loop through all instance ids in instance image
        for instanceId in np.unique(imgNp):
            if instanceId < 1000:
                continue

            instanceObj = Instance(imgNp, instanceId)
            instanceObj_dict = instanceObj.toDict()

            if id2label[instanceObj.labelID].hasInstances:
                mask = (imgNp == instanceId).astype(np.uint8)
                
                _, contour, hier = cv2.findContours(
                    mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                polygons = [c.reshape(-1).tolist() for c in contour]
                instanceObj_dict["contours"] = polygons

            instances[id2label[instanceObj.labelID].name].append(
                instanceObj_dict)
        return instances

    def polygon_to_bbox(polygon: list) -> list:
        """Convert polygon into COCO-format bounding box."""

        # https://github.com/facebookresearch/maskrcnn-benchmark/issues/288#issuecomment-449098063
        TO_REMOVE = 1

        x0 = min(min(p[::2]) for p in polygon)
        x1 = max(max(p[::2]) for p in polygon)
        y0 = min(min(p[1::2]) for p in polygon)
        y1 = max(max(p[1::2]) for p in polygon)

        bbox = [x0, y0, x1 -x0 + TO_REMOVE, y1 - y0 + TO_REMOVE]
        return bbox

    print(src_path)
    src_path = Path(src_path)
    des_path = Path(des_path)
    assert src_path.exists(), "Source annotation file does not exist"
    if des_path.exists():
        print(f"{des_path} exists. Override? (y/n)", end=" ")
        if input() != "y":
            print("Abort")
            return
    else:
        des_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialization
    coco_anno_dict = {
        "images": [],
        "categories": [],
        "annotations": [],
    }
    num_images = 0
    num_categories = 0
    num_annotations = 0

    # Categories
    if car_only:
        categories = ("car",)
    category_to_id = {}
    for category in categories:
        coco_anno_dict["categories"].append({
            "id": num_categories + 1,
            "name": category
        })
        category_to_id[category] = num_categories + 1
        num_categories += 1

    # Start Conversion
    for file in tqdm(list(src_path.rglob("*instanceIds.png"))):
        ##### Images #####
        img_info = {"id": num_images}
        num_images += 1
        img_info["file_name"] = \
            str(file.name).split("_", maxsplit=1)[0] + "/" + \
            str(file.name).replace("gtFine", "leftImg8bit").replace("_instanceIds", "")
        if foggy:
            img_info["file_name"] = \
                img_info["file_name"].replace("leftImg8bit", "leftImg8bit_foggy_beta_0.02")
        with open(str(file).replace("instanceIds.png", "polygons.json"), "r") as f:
            polygon_info = json.load(f)
            img_info["width"] = polygon_info["imgWidth"]
            img_info["height"] = polygon_info["imgHeight"]
        coco_anno_dict["images"].append(img_info)

        ##### Annotations #####
        instances = get_instances_with_polygons(str(file.absolute()))
        for category in instances.keys():
            if category not in categories:
                continue
            for instance in instances[category]:
                anno_info = {
                    "id": num_annotations,
                    "image_id": img_info["id"],
                    "category_id": category_to_id[category],
                    "segmentation": [],
                    "iscrowd": 0,
                    "area": instance["pixelCount"],
                    "bbox": polygon_to_bbox(instance["contours"]),
                }
                num_annotations += 1
                coco_anno_dict["annotations"].append(anno_info)

    print("# of images:", num_images)
    print("# of categories:", num_categories)
    print("# of annotations:", num_annotations)

    with open(des_path, 'w') as f:
        f.write(json.dumps(coco_anno_dict, indent=4))
    print(f"Convert successfully to {des_path}")

def dawn_to_coco(src_path, des_path, car_only, categories: tuple = (
        "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle")
    ) -> None:
    # determine if using txt or XML annotation files 
    ext = ".xml"

    src_path = Path(src_path)
    des_path = Path(des_path)

    assert src_path.exists(), f'Source annotation file {src_path} does not exist'
    if des_path.exists():
        print(f"{des_path} exists. Override? (y/n)", end=" ")
        if input() != "y":
            print("Abort")
            return
    else:
        des_path.parent.mkdir(parents=True, exist_ok=True)

    PATHS = {
        "p": (
                src_path / "Fog", src_path / "Fog/Annotations_XML/", 
                src_path / "Snow", src_path / "Snow/Annotations_XML/", 
                src_path / "Rain", src_path / "Rain/Annotations_XML/", 
                src_path / "Sand", src_path / "Sand/Annotations_XML/"
            ),
    }

    fog_img_folder, fog_ann_file, snow_img_folder, snow_ann_file, rain_img_folder, rain_ann_file, sand_img_folder, sand_ann_file, = PATHS["p"]
    
    # Initialization
    coco_anno_dict = {
        "images": [],
        "categories": [],
        "annotations": [],
    }

    num_images = 0
    num_categories = 0
    num_annotations = 0

    # Categories
    category_to_id = {}
    for category in categories:
        coco_anno_dict["categories"].append({
            "id": num_categories + 1,
            "name": category
        })
        category_to_id[category] = num_categories + 1
        num_categories += 1

    img_paths = []
    img_ann_paths = []

    # read in fog image paths 
    for filename in sorted(os.listdir(fog_img_folder)):
        if filename[-4:] == ".jpg":
            img_paths.append(os.path.join(fog_img_folder, filename))

    # read in fog annotation paths 
    for filename in sorted(os.listdir(fog_ann_file)):
        if filename[-4:] == ext:
            img_ann_paths.append(os.path.join(fog_ann_file, filename))

    # read in snow image paths 
    for filename in sorted(os.listdir(snow_img_folder)):
        if filename[-4:] == ".jpg":
            img_paths.append(os.path.join(snow_img_folder, filename))

    # read in snow annotation paths 
    for filename in sorted(os.listdir(snow_ann_file)):
        if filename[-4:] == ext:
            img_ann_paths.append(os.path.join(snow_ann_file, filename))

    # read in rain image paths 
    for filename in sorted(os.listdir(rain_img_folder)):
        if filename[-4:] == ".jpg":
            img_paths.append(os.path.join(rain_img_folder, filename))

    # read in rain annotation paths 
    for filename in sorted(os.listdir(rain_ann_file)):
        if filename[-4:] == ext:
            img_ann_paths.append(os.path.join(rain_ann_file, filename))

    # read in sand image paths 
    for filename in sorted(os.listdir(sand_img_folder)):
        if filename[-4:] == ".jpg":
            img_paths.append(os.path.join(sand_img_folder, filename))

    # read in sand annotation paths 
    for filename in sorted(os.listdir(sand_ann_file)):
        if filename[-4:] == ext:
            img_ann_paths.append(os.path.join(sand_ann_file, filename))

    print("image total: " + str(len(img_paths)) + " annotation total: " + str(len(img_ann_paths)))

    for idx in range(len(img_paths)):
        file_name = img_paths[idx]

        ind = -1
        for i in range(len(file_name) - 1, 0, -1):
            if file_name[i] == '/':
                ind = i 
                break 
        # print(file_name[ind + 1:])

        img = Image.open(file_name)
        # print(os.path.join(src_path, file_name[ind + 1:]))
        img.save(os.path.join(src_path, file_name[ind + 1:]), 'JPEG')

        # resize if too many dims 
        # if img.shape[0] == 4:
        #     img = img[:3]

        # print(str(file_name) + " " + str(img.shape))

        # h, w = img.shape[1], img.shape[2]
        w, h = img.size

        ##### Images #####
        img_info = {
            "id": num_images,
            "file_name": file_name,
            "height": h,
            "width": w
        }
        coco_anno_dict["images"].append(img_info)
        num_images += 1

        ##### Annotations #####
        file = open(img_ann_paths[idx], "r")
        tree = ET.parse(file)
        root = tree.getroot()
        for obj in root.findall("object"):
            label = obj.find('name').text
            box = obj.find('bndbox')
            x1 = float(box.find("xmin").text)
            x2 = float(box.find("xmax").text)
            y1 = float(box.find("ymin").text)
            y2 = float(box.find("ymax").text)

            # b = box_xyxy_to_cxcywh(torch.tensor([xmin, ymin, xmax, ymax]))
            # b = b / torch.tensor([w, h, w, h], dtype=torch.float32)
            # boxes.append(b.tolist())
            # labels.append(self.class_to_ind[obj_class])

            # print(label)
            if label not in category_to_id:
                continue
            anno_info = {
                "id": num_annotations,
                "image_id": img_info["id"],
                "category_id": category_to_id[label],
                "segmentation": [],
                "iscrowd": 0,
            }
            num_annotations += 1

            # Bbox
            anno_info["bbox"] = [x1, y1, x2 - x1, y2 - y1]
            anno_info["area"] = float((x2 - x1) * (y2 - y1))
            coco_anno_dict["annotations"].append(anno_info)

    print("# of images:", num_images)
    print("# of categories:", num_categories)
    print("# of annotations:", num_annotations)

    with open(des_path, 'w') as f:
        f.write(json.dumps(coco_anno_dict, indent=4))
    print(f"Convert successfully to {des_path}")

if __name__ == "__main__":
    # sim10k_to_coco(
    #     src_path="VOC2012/Annotations",
    #     des_path="annotations/sim10k_caronly.json"
    # )

    # bdd100k_daytime_to_coco(
    #     src_path="labels/bdd100k_labels_images_train.json",
    #     des_path="annotations/bdd_daytime_train.json"
    # )
    # bdd100k_daytime_to_coco(
    #     src_path="labels/bdd100k_labels_images_val.json",
    #     des_path="annotations/bdd_daytime_val.json"
    # )

    # cityscapes_to_coco(
    #     src_path="/opt/DA_Object_Detection/AQT/datasets/cityscapes/gtFine/train",
    #     des_path="/opt/DA_Object_Detection/AQT/datasets/cityscapes/annotations/cityscapes_train.json",
    # )
    # cityscapes_to_coco(
    #     src_path="/opt/DA_Object_Detection/AQT/datasets/cityscapes/gtFine/val",
    #     des_path="/opt/DA_Object_Detection/AQT/datasets/cityscapes/annotations/cityscapes_val.json",
    # )
    # cityscapes_to_coco(
    #     src_path="/opt/DA_Object_Detection/AQT/datasets/cityscapes/gtFine/train",
    #     des_path="/opt/DA_Object_Detection/AQT/datasets/cityscapes/annotations/cityscapes_caronly_train.json",
    #     car_only=True,
    # )
    # cityscapes_to_coco(
    #     src_path="/opt/DA_Object_Detection/AQT/datasets/cityscapes/gtFine/val",
    #     des_path="/opt/DA_Object_Detection/AQT/datasets/cityscapes/annotations/cityscapes_caronly_val.json",
    #     car_only=True,
    # )
    # cityscapes_to_coco(
    #     src_path="/opt/DA_Object_Detection/AQT/datasets/cityscapes/gtFine/train",
    #     des_path="/opt/DA_Object_Detection/AQT/datasets/cityscapes/annotations/foggy_cityscapes_train.json",
    #     foggy=True,
    # )
    # cityscapes_to_coco(
    #     src_path="/opt/DA_Object_Detection/AQT/datasets/cityscapes/gtFine/val",
    #     des_path="/opt/DA_Object_Detection/AQT/datasets/cityscapes/annotations/foggy_cityscapes_val.json",
    #     foggy=True,
    # )

    dawn_to_coco(
        "/opt/MIC/datasets/DAWN_2/Train",
        "/opt/MIC/datasets/DAWN_2/annotations/dawn_train.json",
        False
    )

    dawn_to_coco(
        "/opt/MIC/datasets/DAWN_2/Val",
        "/opt/MIC/datasets/DAWN_2/annotations/dawn_val.json",
        False
    )