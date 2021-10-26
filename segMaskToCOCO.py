from PIL import Image
import json
import os
from timeit import default_timer as timer

BACKGROUND_COLOR = (0, 0, 0)  # black
root = '../../Output/train'


def rle(pixelMatrix, width, height):
	lastColor = BACKGROUND_COLOR
	tmpDict = {lastColor: {"counts": [0], "area": 0}}
	bboxDict = {
	    lastColor: {
	        "xmin": 0,
	        "xmax": width - 1,
	        "ymin": 0,
	        "ymax": height - 1,
	    }
	}

	counter = 0
	for x in range(width):
		for y in range(height):
			#i = (y * width) + x
			clr = pixelMatrix[x, y]

			if lastColor != clr:
				tmpDict[lastColor]["counts"].append(0)
				bboxDict[lastColor]["xmax"] = max(bboxDict[lastColor]["xmax"], x)
				bboxDict[lastColor]["ymax"] = max(bboxDict[lastColor]["ymax"], y - 1)

				if clr in tmpDict:
					tmpDict[clr]["counts"].append(0)
					bboxDict[clr]["xmin"] = min(bboxDict[clr]["xmin"], x)
					bboxDict[clr]["ymin"] = min(bboxDict[clr]["ymin"], y)

				else:
					tmpDict[clr] = {}
					tmpDict[clr]["counts"] = [counter, 0]
					tmpDict[clr]["area"] = 0
					bboxDict[clr] = {
					    "xmin": x,
					    "xmax": 0,
					    "ymin": y,
					    "ymax": 0,
					}

			for key in tmpDict:
				tmpDict[key]["counts"][-1] += 1

			lastColor = clr
			counter += 1
			tmpDict[lastColor]["area"] += 1

	return (tmpDict, bboxDict)


def UpdateAnnotations(data, imgMatrix, width, height, outFilename):
	thisTime = timer()
	(rleDict, bboxDict) = rle(imgMatrix, width, height)
	for color in rleDict:
		if color != BACKGROUND_COLOR:
			data["annotations"].append({
			    "id": len(data["annotations"]),
			    "image_id": len(data["images"]),
			    "category_id": 1,
			    "segmentation": {
			        "counts": rleDict[color]["counts"],
			        "size": [height, width]
			    },
			    "iscrowd": 0,
			    "area": rleDict[color]["area"],
			    "bbox": [bboxDict[color]["xmin"], bboxDict[color]["ymin"], bboxDict[color]["xmax"] - bboxDict[color]["xmin"], bboxDict[color]["ymax"] - bboxDict[color]["ymin"]]
			})

	data["images"].append({"id": len(data["images"]), "file_name": outFilename, "width": width, "height": height})
	print(f"OK [{timer() - thisTime:.4f}s]\t{outFilename}")


def SegmentToCOCO(path):
	categories = [{
	    "supercategory": "person",
	    "id": 1,
	    "name": "person"
	}, {
	    "supercategory": "vehicle",
	    "id": 2,
	    "name": "bicycle"
	}, {
	    "supercategory": "vehicle",
	    "id": 3,
	    "name": "car"
	}, {
	    "supercategory": "vehicle",
	    "id": 4,
	    "name": "motorcycle"
	}, {
	    "supercategory": "vehicle",
	    "id": 5,
	    "name": "airplane"
	}, {
	    "supercategory": "vehicle",
	    "id": 6,
	    "name": "bus"
	}, {
	    "supercategory": "vehicle",
	    "id": 7,
	    "name": "train"
	}, {
	    "supercategory": "vehicle",
	    "id": 8,
	    "name": "truck"
	}, {
	    "supercategory": "vehicle",
	    "id": 9,
	    "name": "boat"
	}, {
	    "supercategory": "outdoor",
	    "id": 10,
	    "name": "traffic light"
	}, {
	    "supercategory": "outdoor",
	    "id": 11,
	    "name": "fire hydrant"
	}, {
	    "supercategory": "outdoor",
	    "id": 13,
	    "name": "stop sign"
	}, {
	    "supercategory": "outdoor",
	    "id": 14,
	    "name": "parking meter"
	}, {
	    "supercategory": "outdoor",
	    "id": 15,
	    "name": "bench"
	}, {
	    "supercategory": "animal",
	    "id": 16,
	    "name": "bird"
	}, {
	    "supercategory": "animal",
	    "id": 17,
	    "name": "cat"
	}, {
	    "supercategory": "animal",
	    "id": 18,
	    "name": "dog"
	}, {
	    "supercategory": "animal",
	    "id": 19,
	    "name": "horse"
	}, {
	    "supercategory": "animal",
	    "id": 20,
	    "name": "sheep"
	}, {
	    "supercategory": "animal",
	    "id": 21,
	    "name": "cow"
	}, {
	    "supercategory": "animal",
	    "id": 22,
	    "name": "elephant"
	}, {
	    "supercategory": "animal",
	    "id": 23,
	    "name": "bear"
	}, {
	    "supercategory": "animal",
	    "id": 24,
	    "name": "zebra"
	}, {
	    "supercategory": "animal",
	    "id": 25,
	    "name": "giraffe"
	}, {
	    "supercategory": "accessory",
	    "id": 27,
	    "name": "backpack"
	}, {
	    "supercategory": "accessory",
	    "id": 28,
	    "name": "umbrella"
	}, {
	    "supercategory": "accessory",
	    "id": 31,
	    "name": "handbag"
	}, {
	    "supercategory": "accessory",
	    "id": 32,
	    "name": "tie"
	}, {
	    "supercategory": "accessory",
	    "id": 33,
	    "name": "suitcase"
	}, {
	    "supercategory": "sports",
	    "id": 34,
	    "name": "frisbee"
	}, {
	    "supercategory": "sports",
	    "id": 35,
	    "name": "skis"
	}, {
	    "supercategory": "sports",
	    "id": 36,
	    "name": "snowboard"
	}, {
	    "supercategory": "sports",
	    "id": 37,
	    "name": "sports ball"
	}, {
	    "supercategory": "sports",
	    "id": 38,
	    "name": "kite"
	}, {
	    "supercategory": "sports",
	    "id": 39,
	    "name": "baseball bat"
	}, {
	    "supercategory": "sports",
	    "id": 40,
	    "name": "baseball glove"
	}, {
	    "supercategory": "sports",
	    "id": 41,
	    "name": "skateboard"
	}, {
	    "supercategory": "sports",
	    "id": 42,
	    "name": "surfboard"
	}, {
	    "supercategory": "sports",
	    "id": 43,
	    "name": "tennis racket"
	}, {
	    "supercategory": "kitchen",
	    "id": 44,
	    "name": "bottle"
	}, {
	    "supercategory": "kitchen",
	    "id": 46,
	    "name": "wine glass"
	}, {
	    "supercategory": "kitchen",
	    "id": 47,
	    "name": "cup"
	}, {
	    "supercategory": "kitchen",
	    "id": 48,
	    "name": "fork"
	}, {
	    "supercategory": "kitchen",
	    "id": 49,
	    "name": "knife"
	}, {
	    "supercategory": "kitchen",
	    "id": 50,
	    "name": "spoon"
	}, {
	    "supercategory": "kitchen",
	    "id": 51,
	    "name": "bowl"
	}, {
	    "supercategory": "food",
	    "id": 52,
	    "name": "banana"
	}, {
	    "supercategory": "food",
	    "id": 53,
	    "name": "apple"
	}, {
	    "supercategory": "food",
	    "id": 54,
	    "name": "sandwich"
	}, {
	    "supercategory": "food",
	    "id": 55,
	    "name": "orange"
	}, {
	    "supercategory": "food",
	    "id": 56,
	    "name": "broccoli"
	}, {
	    "supercategory": "food",
	    "id": 57,
	    "name": "carrot"
	}, {
	    "supercategory": "food",
	    "id": 58,
	    "name": "hot dog"
	}, {
	    "supercategory": "food",
	    "id": 59,
	    "name": "pizza"
	}, {
	    "supercategory": "food",
	    "id": 60,
	    "name": "donut"
	}, {
	    "supercategory": "food",
	    "id": 61,
	    "name": "cake"
	}, {
	    "supercategory": "furniture",
	    "id": 62,
	    "name": "chair"
	}, {
	    "supercategory": "furniture",
	    "id": 63,
	    "name": "couch"
	}, {
	    "supercategory": "furniture",
	    "id": 64,
	    "name": "potted plant"
	}, {
	    "supercategory": "furniture",
	    "id": 65,
	    "name": "bed"
	}, {
	    "supercategory": "furniture",
	    "id": 67,
	    "name": "dining table"
	}, {
	    "supercategory": "furniture",
	    "id": 70,
	    "name": "toilet"
	}, {
	    "supercategory": "electronic",
	    "id": 72,
	    "name": "tv"
	}, {
	    "supercategory": "electronic",
	    "id": 73,
	    "name": "laptop"
	}, {
	    "supercategory": "electronic",
	    "id": 74,
	    "name": "mouse"
	}, {
	    "supercategory": "electronic",
	    "id": 75,
	    "name": "remote"
	}, {
	    "supercategory": "electronic",
	    "id": 76,
	    "name": "keyboard"
	}, {
	    "supercategory": "electronic",
	    "id": 77,
	    "name": "cell phone"
	}, {
	    "supercategory": "appliance",
	    "id": 78,
	    "name": "microwave"
	}, {
	    "supercategory": "appliance",
	    "id": 79,
	    "name": "oven"
	}, {
	    "supercategory": "appliance",
	    "id": 80,
	    "name": "toaster"
	}, {
	    "supercategory": "appliance",
	    "id": 81,
	    "name": "sink"
	}, {
	    "supercategory": "appliance",
	    "id": 82,
	    "name": "refrigerator"
	}, {
	    "supercategory": "indoor",
	    "id": 84,
	    "name": "book"
	}, {
	    "supercategory": "indoor",
	    "id": 85,
	    "name": "clock"
	}, {
	    "supercategory": "indoor",
	    "id": 86,
	    "name": "vase"
	}, {
	    "supercategory": "indoor",
	    "id": 87,
	    "name": "scissors"
	}, {
	    "supercategory": "indoor",
	    "id": 88,
	    "name": "teddy bear"
	}, {
	    "supercategory": "indoor",
	    "id": 89,
	    "name": "hair drier"
	}, {
	    "supercategory": "indoor",
	    "id": 90,
	    "name": "toothbrush"
	}]
	data = {"info": {"description": "Unity Crowd dataset"}, "images": [], "annotations": [], "categories": categories}

	for root, dirs, files in os.walk(path):
		for file in files:
			if file.endswith("_mask.bmp"):
				filename = os.path.join(root, file)
				outFilename = os.path.basename(root) + '/' + file.replace("_mask", "")
				im = Image.open(filename)
				pix = im.load()
				UpdateAnnotations(
				    data,
				    pix,
				    im.size[0],
				    im.size[1],
				    outFilename,
				)

	with open(path + '/output.json', 'w') as outfile:
		json.dump(data, outfile, indent=2)


if __name__ == '__main__':
	print("START")
	start = timer()
	SegmentToCOCO(root)
	end = timer()
	print(f"DONE in {end:.2f}s")
