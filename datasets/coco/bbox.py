import json
from pycocotools.coco import COCO
import os

# Creates new COCO json that does not include images with people that have a very small BBOX
# change paths according to your setup

def cleanBbox(js, coco, percentage):

	annotations = []
	images = []
	images_to_delete = []
	for image in js["images"]:
		id = image["id"]
		W = image["width"]
		H = image["height"]
		dim = H * W

		annIds = coco.getAnnIds(imgIds=[id])
		anns = coco.loadAnns(annIds)

		annotationsList = []
		keep = True
		for ann in anns:
			bbox = ann["bbox"]
			w = bbox[2]
			h = bbox[3]
			annDim = h * w
			perc = annDim * 100 / dim
			if perc > percentage:
				annotationsList.append(ann)
			else:
				keep = False

		if keep:
			annotations.extend(annotationsList)
			images.append(image)
		else:
			images_to_delete.append(image["file_name"])

	js["annotations"] = annotations
	js["images"] = images
	with open("./datasets/coco/annotations/cleanBboxAnn.json", "w+") as output:
		json.dump(js, output)

	for image in images_to_delete:
		file_path = f'./datasets/coco/testReal_bbox_filtered/{image}'
		if os.path.isfile(file_path):
			os.remove(file_path)
		else:
			print("image not in folder")


percentage = 1
annotationFile = annotationFile = "./datasets/coco/annotations/persons.json"
coco = COCO(annotationFile)

with open(annotationFile, "r+") as input:
	input = json.load(input)
	cleanBbox(input, coco, percentage)