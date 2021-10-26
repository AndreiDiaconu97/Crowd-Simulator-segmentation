from pycocotools.coco import COCO
import numpy as np
import json
import os

# Creates new COCO json that does not include images with too few people
# change paths according to your setup

annFile = './annotations/instances_train2017.json'
imgFolder = './testReal'
imgNames = os.listdir(imgFolder)

coco = COCO(annFile)

catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds=catIds)
personImg = []
images_to_delete = []
imagesId = []

with open(annFile, "r+") as data_file:
	with open("./annotations/persons.json", "w+") as output_file:
		print("Collecting JSON")

		data = json.load(data_file)
		totalImgs = len(data["images"])

		counter = 0
		for image in data["images"]:
			counter += 1
			print("processing {} image, {} total  -  {} %".format(counter, totalImgs, (counter * 100 / totalImgs)))

			if (image["id"] in imgIds and image["file_name"] in imgNames):
				id = image["id"]
				image_anns = coco.getAnnIds(catIds=catIds, imgIds=id)
				if (len(image_anns) < 5):
					images_to_delete.append(image["file_name"])
				else:
					imagesId.append(id)
			else:
				images_to_delete.append(image["file_name"])

		print("Done collecting Images")

		annIds = coco.getAnnIds(catIds=catIds, imgIds=imagesId)
		anns = coco.loadAnns(annIds)
		personImg = coco.loadImgs(imagesId)
		data["annotations"] = anns
		data["images"] = personImg

		print("Writing on file")

		json.dump(data, output_file)

print("Deleting images")

for image in images_to_delete:
	file_path = '{}/{}'.format(imgFolder, image)
	if os.path.isfile(file_path):
		os.remove(file_path)
	else:
		print("image not in folder")
