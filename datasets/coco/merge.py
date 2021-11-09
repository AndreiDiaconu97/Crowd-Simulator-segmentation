import json
import pycocotools.mask as mask
from pycocotools.coco import COCO
import numpy as np
import json
import numpy as np
from pycocotools import mask
from skimage import measure


def mask2Pol(mask):
    '''
    Auxiliary function for convertJson
    '''
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, 0)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons


def close_contour(contour):
    '''
    Auxiliary function for convertJson
    '''
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def rle2Mask(ann, h, w):
    '''
    Auxiliary function for convertJson
    '''
    segmentation = ann["segmentation"]
    rle = mask.frPyObjects(segmentation, h, w)
    bitmap_mask = np.array(mask.decode(rle), np.uint8)
    bitmap_mask = (bitmap_mask * 255).astype("uint8")

    return bitmap_mask


def convertJson(js):
    '''
    Given an annotation file were the the segmentation mask are in RLE uncompressed format, it convert each annotation to polygon file and write the result to file
    '''
    annotations = []
    totalAnns = len(js["annotations"])
    toatlImages = len(js["images"])
    counter = 0
    for ann in js["annotations"]:
        counter += 1
        print(f"[INFO] Parsing {counter} annotation over {totalAnns} total. {round(counter * 100 /totalAnns)}%")
        # ann = js["annotations"][0]
        image = js["images"]["id" == ann["image_id"]]
        h = image["height"]
        w = image["width"]

        bitmask = rle2Mask(ann, h, w)
        pol = mask2Pol(bitmask)

        ann["segmentation"] = pol
        annotations.append(ann)

    js["annotations"] = annotations
    with open("./converted.json", "w+") as file:
        print("[INFO] saving copy of converted Json...")
        json.dump(js, file)
        print("[INFO] Done")

    return js, toatlImages


def reduceAnn(js, n, k, coco):
    '''
    Auxiliary function for merge
    '''
    annotations = []
    images = []
    counter = 0
    for image in js["images"]:
        counter += 1
        id = image["id"]

        if counter > k:
            break

        print(f"[INFO] Parsing {counter} image over {k} total. {round(counter * 100 /k)}%")
        annIds = coco.getAnnIds(imgIds=[id])
        anns = coco.loadAnns(annIds)
        image["id"] = id + n
        image["file_name"] = "train2017\\" + image["file_name"]
        images.append(image)
        for ann in anns:
            ann["image_id"] = id + n
            ann["id"] = ann["id"] + n
            annotations.append(ann)

    js["annotations"] = annotations
    js["images"] = images
    with open("./reduced.json", "w+") as file:
        print("[INFO] saving copy of reduced Json...")
        json.dump(js, file)
        print("[INFO] Done")


def merge(json1, json2):
    '''
    Given two annotations files, it redued the number of annotation of the second file to match the first. It then merge them and write the product to file
    '''
    with open(json1, "r+") as file1:
        with open(json2, "r+") as file2:
            with open("test.json", "w+") as output:

                print("[INFO] Loading files...")
                dictA = json.load(file1)
                dictB = json.load(file2)
                print("[INFO] Done")

                annotationsA = len(dictA["annotations"])
                imagesA = len(dictA["images"])

                print("[INFO] Reducing number of annotations...")
                coco = COCO(json2)
                reduceAnn(dictB, annotationsA, imagesA, coco)
                annotationsB = len(dictB["annotations"])

                print("[INFO] Merging Files...")
                info = dictA["info"]
                licenses = dictB["licenses"]
                images = [l for l in dictA["images"] + dictB["images"]]
                annotations = [l for l in dictA["annotations"] + dictB["annotations"]]
                categories = dictA["categories"]

                merged_dict = {"info": info, "licences": licenses, "images": images, "annotations": annotations, "categories": categories}
                print("[INFO] Done")

                annotationsFinal = len(merged_dict["annotations"])
                print(f"[INFO] Total number of annotations: {annotationsFinal} should be {annotationsA + annotationsB}")

                print("[INFO] Writing output...")
                json.dump(merged_dict, output)
                print("[INFO] Done")
