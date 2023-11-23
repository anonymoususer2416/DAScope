# ----------------------------------------------
# Created by Wei-Jie Huang
# A collection of annotation conversion scripts
# We provide this for reference only
# ----------------------------------------------


import json
from pathlib import Path
from tqdm import tqdm

AREAS = {
    "extremely_small"  : [0, 144],
    "relatively_small" : [144, 400],
    "generally_small"  : [400, 1024],
    "normal"           : [1024, 2000],
}

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

def size_check(pixel_area):
    for key in AREAS.keys():
        rang = AREAS[key]
        if pixel_area >= rang[0] and pixel_area <= rang[1]:
            return key

    return "ignore"

def bdd100k_daytime_to_coco(
    src_path: str = "labels/bdd100k_labels_images_train.json",
    des_path: str = "annotations/bdd_daytime_train.json",
    save_data: bool = True,
    small_only: bool = False,
    size_calc: bool = False,
    categories: tuple = (
        "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle")
    ) -> None:

    r""" Extract ``daytime`` subset from BDD100k dataset and convert into COCO format.
    Args:
        src_path: source of the annotation json file
        des_path: destination of the converted COCO-fomat annotation
        categories: categories used
    """
    cat_dict = {}
    for c in categories:
        cat_dict[c] = 0

    src_path = Path(src_path)
    des_path = Path(des_path)
    assert src_path.exists(), "Source annotation file does not exist"
    if des_path.exists() and save_data == True:
        print(f"{des_path} exists. Override? (y/n)", end=" ")
        if input() != "y":
            print("Abort")
            return
    else:
        des_path.parent.mkdir(parents=True, exist_ok=True)

    if size_calc:
        area_counts = {
            "extremely_small"  : 0,
            "generally_small"  : 0,
            "relatively_small" : 0,
            "normal"           : 0,
            "ignore"           : 0,
        }
        num_imgs_nosmall = 0
        
        cat_2_sizes = {}
        for c in categories:
            cat_2_sizes[c] = area_counts.copy()

    # Initialization
    coco_anno_dict = {
        "images": [],
        "categories": [],
        "annotations": [],
    }
    num_images = 0
    num_categories = 0
    num_annotations = 0
    
    total_labels = {}

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
        num_additions = 0
        for label in raw_img_anno["labels"]:
            if label["category"] == "bike":
                label["category"] = "bicycle"

            if label["category"] == "motor":
                label["category"] = "motorcycle"
                
            if label["category"] not in total_labels:
                total_labels[label["category"]] = 1
            else:
                total_labels[label["category"]] += 1
            
            if label["category"] not in category_to_id or "box2d" not in label or label["category"] == "train":
                continue
            
            anno_info = {
                "id": num_annotations,
                "image_id": img_info["id"],
                "category_id": category_to_id[label["category"]],
                "segmentation": [],
                "iscrowd": 0,
            }

            # Bbox
            x1 = label["box2d"]["x1"]
            y1 = label["box2d"]["y1"]
            x2 = label["box2d"]["x2"]
            y2 = label["box2d"]["y2"]
            area = float((x2 - x1) * (y2 - y1))

            key = None
            if size_calc:
                key = size_check(area)
                area_counts[key] += 1
                cat_2_sizes[label["category"]][key] += 1

            add = True
            if small_only and key == "ignore":
                add = False

            if add:
                num_additions += 1
                num_annotations += 1
                anno_info["bbox"] = [x1, y1, x2 - x1, y2 - y1]
                anno_info["area"] = area
                coco_anno_dict["annotations"].append(anno_info)
                cat_dict[label["category"]] += 1

            if num_additions == 0:
                num_imgs_nosmall += 1

    print("=========================PRINTING BDD100K STATISTICS=========================")
    print("# of images: ", num_images)
    print("# of categories: ", num_categories)
    print("# of annotations/objects: ", num_annotations)
    for category in total_labels:
        print(category + ": " + str(total_labels[category]) + " number of objects: " + str(total_labels[category]))
    
    for category in cat_dict:
        print(category + ": " + str(cat_dict[category]) + " percent of objects: " + str(cat_dict[category] / num_annotations))
        
        if size_calc == True:
            for key in cat_2_sizes[category]:
                print(category + " number of " + str(key) + " sized objects: " + str(cat_2_sizes[category][key]))
                
    if size_calc == True:
        print("# of images with no small objects: " + str(num_imgs_nosmall))
        for size in area_counts:
            print(size + ": " + str(area_counts[size]))

    if save_data == True:
        with open(des_path, 'w') as f:
            f.write(json.dumps(coco_anno_dict, indent=4))
        print(f"Convert successfully to {des_path}")


def cityscapes_to_coco(
    src_path: str = "gtFine/train",
    des_path: str = "annotations/cityscapes_train.json",
    save_data: bool = True,
    small_only: bool = False,
    size_calc: bool = False,
    car_only: bool = False,
    foggy: bool = False,
    defog: bool = False,
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
    cat_dict = {}
    for c in categories:
        cat_dict[c] = 0
        
    total_labels = {}

    if size_calc:
        area_counts = {
            "extremely_small"  : 0,
            "generally_small"  : 0,
            "relatively_small" : 0,
            "normal"           : 0,
            "ignore"           : 0,
        }
        num_imgs_nosmall = 0
        
        cat_2_sizes = {}
        for c in categories:
            cat_2_sizes[c] = area_counts.copy()

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
                
                contour, hier = cv2.findContours(
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
    if des_path.exists() and save_data == True:
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
        if defog:
             img_info["file_name"] = \
                img_info["file_name"].replace("leftImg8bit", "leftImg8bit_defog_beta_0.02")
        with open(str(file).replace("instanceIds.png", "polygons.json"), "r") as f:
            polygon_info = json.load(f)
            img_info["width"] = polygon_info["imgWidth"]
            img_info["height"] = polygon_info["imgHeight"]
        coco_anno_dict["images"].append(img_info)

        ##### Annotations #####
        instances = get_instances_with_polygons(str(file.absolute()))
        for category in instances.keys():
            if category == "train":
                continue    
            
            if category not in total_labels:
                total_labels[category] = 1
            else:
                total_labels[category] += 1
                
            if category not in categories:
                continue

            # print(len(instances[category]))
            num_additions = 0
            for instance in instances[category]:
                key = None
                if size_calc:
                    key = size_check(instance["pixelCount"])
                    area_counts[key] += 1
                    cat_2_sizes[category][key] += 1
                
                anno_info = {
                    "id": num_annotations,
                    "image_id": img_info["id"],
                    "category_id": category_to_id[category],
                    "segmentation": [],
                    "iscrowd": 0,
                    "area": instance["pixelCount"],
                    "bbox": polygon_to_bbox(instance["contours"]),
                }

                add = True
                if small_only and key == "ignore":
                    add = False

                if add:
                    num_additions += 1
                    num_annotations += 1
                    coco_anno_dict["annotations"].append(anno_info)
                    cat_dict[category] += 1

                if num_additions == 0:
                    num_imgs_nosmall += 1

    print("=========================PRINTING CITYSCAPES STATISTICS=========================")
    print("# of images:", num_images)
    print("# of categories:", num_categories)
    print("# of annotations/objects:", num_annotations)
    for category in total_labels:
        print(category + ": " + str(total_labels[category]) + " number of objects: " + str(total_labels[category]))
        
    for category in cat_dict:
        print(category + ": " + str(cat_dict[category]) + " percent of objects: " + str(cat_dict[category] / num_annotations))
        
        if size_calc == True:
            for key in cat_2_sizes[category]:
                print(category + " number of " + str(key) + " sized objects: " + str(cat_2_sizes[category][key]))

    if size_calc == True:
        print("# of images with no small objects: " + str(num_imgs_nosmall))
        for size in area_counts:
            print(size + ": " + str(area_counts[size]))

    if save_data == True:
        with open(des_path, 'w') as f:
            f.write(json.dumps(coco_anno_dict, indent=4))
        print(f"Convert successfully to {des_path}")

# To use anno_convert.py to convert a dataset into coco format, simply uncomment the conversion method you want to run below.
if __name__ == "__main__":
    # bdd100k_daytime_to_coco(
    #     src_path="/opt/TDD/datasets/bdd_daytime/labels/bdd100k_labels_images_train.json",
    #     des_path="/opt/TDD/datasets/bdd_daytime/annotations/bdd_daytime_train_small.json",
    #     save_data=False,
    #     small_only=True,
    #     size_calc=True
    # )
    # bdd100k_daytime_to_coco(
    #     src_path="/opt/TDD/datasets/bdd_daytime/labels/bdd100k_labels_images_val.json",
    #     des_path="/opt/TDD/datasets/bdd_daytime/annotations/bdd_daytime_val_small.json",
    #     save_data=False,
    #     small_only=True,
    #     size_calc=True
    # )

    cityscapes_to_coco(
        src_path="/opt/MIC/datasets/cityscapes/gtFine/train",
        des_path="/opt/MIC/datasets/cityscapes/annotations/cityscapes_train_small.json",
        save_data=False,
        small_only=False,
        size_calc=True,
    )
    cityscapes_to_coco(
        src_path="/opt/MIC/datasets/cityscapes/gtFine/val",
        des_path="/opt/MIC/datasets/cityscapes/annotations/cityscapes_val_small.json",
        save_data=False,
        small_only=False,
        size_calc=True
    )

    # cityscapes_to_coco(
    #     src_path="/opt/MIC/datasets/cityscapes/gtFine/train",
    #     des_path="/opt/MIC/datasets/cityscapes/annotations/foggy_cityscapes_train_small.json",
    #     save_data=True,
    #     small_only=True,
    #     size_calc=True,
    #     foggy=True,
    #     defog=False,
    # )
    # cityscapes_to_coco(
    #     src_path="/opt/MIC/datasets/cityscapes/gtFine/val",
    #     des_path="/opt/MIC/datasets/cityscapes/annotations/foggy_cityscapes_val_small.json",
    #     save_data=True,
    #     small_only=True,
    #     size_calc=True,
    #     foggy=True,
    #     defog=False,
    # )