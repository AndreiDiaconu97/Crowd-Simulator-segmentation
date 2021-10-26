import cv2
import json
import os
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, DatasetMapper
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.model_zoo.model_zoo import get_config, get
from detectron2.modeling import build_model
from detectron2.utils.visualizer import Visualizer
# from LossEvalHook import LossEvalHook

## SETTINGS ########################################################################

OUT_DIR = "./runs/run_tmp"
PRED_INPUT_FOLDER = "pred_img"
BASE_MODEL = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
# BASE_MODEL = "Misc/scratch_mask_rcnn_R_50_FPN_3x_gn.yaml"

trainDatasetPath = "C:/Users/Adi/Documents/Programming/Crowd-Simulator/Output/train"
# valDatasetPath = "C:/Users/Adi/Documents/Programming/Crowd-Simulator/Output/val"
# testDatasetPath = "./datasets/coco/testSynth"
testDatasetPath = "./datasets/coco/testReal"
# testDatasetPath = "./datasets/coco/testReal_bbox_filtered"

trainAnnotPath = trainDatasetPath + "/output.json"
# valAnnotPath = valDatasetPath + "/output.json"
# testAnnotPath = testDatasetPath + "/output.json"
testAnnotPath = "./datasets/coco/annotations/persons.json"
# testAnnotPath = "./datasets/coco/annotations/cleanBboxAnn.json"

DO_TRAIN = True  # see InitCfg() for custom training settings
RESUME_CHECKPOINT = False  # True for last checkpoint if available, set to False and change cfg.MODEL.WEIGHTS for custom checkpoints
DO_INFERENCE = False
DO_TEST = False

####################################################################################


def VisualizeAnnotations(dataName, imgIndex, toSave):
	dataset_train_metadata = MetadataCatalog.get(dataName)
	dataset_dicts = DatasetCatalog.get(dataName)
	imgDict = dataset_dicts[imgIndex]
	img = cv2.imread(imgDict["file_name"])
	visualizer = Visualizer(img[:, :, ::-1], metadata=dataset_train_metadata, scale=1)
	vis = visualizer.draw_dataset_dict(imgDict)
	out_img = vis.get_image()[:, :, ::-1]
	if toSave:
		cv2.imwrite(f"{OUT_DIR}/input_example_{imgIndex}.png", out_img)
	else:
		cv2.imshow("image", out_img)
		cv2.waitKey(0)


def VisualizePredictions(cfg, thresh, imgPath, toSave):
	origThresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh
	predictor = DefaultPredictor(cfg)
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = origThresh

	dataset_train_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
	img = cv2.imread(imgPath)
	outputs = predictor(img)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
	visualizer = Visualizer(img[:, :, ::-1], metadata=dataset_train_metadata, scale=1)
	vis = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
	out_img = vis.get_image()[:, :, ::-1]
	if toSave:
		cv2.imwrite(OUT_DIR + f"/inference_{os.path.basename(imgPath)}", out_img)
	else:
		cv2.imshow("image", out_img)
		cv2.waitKey(0)


def InitCfg(trainName, testName, trained):
	cfg = get_config(BASE_MODEL, trained=trained)
	cfg.SEED = 42
	cfg.OUTPUT_DIR = OUT_DIR
	# cfg.DATALOADER.NUM_WORKERS = 4
	# cfg.CUDNN_BENCHMARK = True # improves what?
	cfg.DATASETS.TEST = (testName, )
	cfg.DATASETS.TRAIN = (trainName, )
	cfg.INPUT.MASK_FORMAT = "bitmask"
	cfg.MODEL.BACKBONE.FREEZE_AT = 0
	cfg.MODEL.DEVICE = "cuda"
	cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (128)  # faster, and good enough for this toy dataset (default: 512)
	cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 3 classes (data, fig, hazelnut)
	# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # WARNING: influences evaluation!!!
	cfg.SOLVER.BASE_LR = 0.00125  # ERROR if too big: "early training has diverged"
	cfg.SOLVER.IMS_PER_BATCH = 1
	cfg.SOLVER.MAX_ITER = (50000)  # 300 iterations seems good enough, but you can certainly train longer
	cfg.SOLVER.STEPS = [40000, 48000]  # do not decay learning rate
	# cfg.SOLVER.WARMUP_ITERS = 100
	cfg.TEST.EVAL_PERIOD = 1000  # validation rate
	cfg.SOLVER.CHECKPOINT_PERIOD = 2500  # model autosave rate
	return cfg


class MyTrainer(DefaultTrainer):
	@classmethod
	def build_evaluator(cls, cfg, dataset_name, output_folder=None):
		if output_folder is None:
			output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
		return COCOEvaluator(dataset_name, cfg, True, output_folder)

	def build_hooks(self):
		hooks = super().build_hooks()
		# hooks.insert(-1, LossEvalHook(self.cfg.TEST.EVAL_PERIOD, self.model, build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0], DatasetMapper(self.cfg, True))))
		return hooks


def LoadTrainer(cfg, resumeTraining):
	# bool resumeTraining :
	# if True	=> load whole state	 from .pth referenced by cfg.OUTPUT_DIR/last_checkpoint file, and continue training from there
	# if False	=> load weights file from .pth referenced by cfg.MODEL.WEIGHTS path and start new training
	# WARNING: True is ignored if last_checkpoint does not exist
	trainer = MyTrainer(cfg)
	trainer.resume_or_load(resume=resumeTraining)
	return trainer


def SaveCfg(cfg):
	with open(f"{OUT_DIR}/cfg.yaml", "w") as f:
		f.write(cfg.dump())


def main():
	trainDatasetName = "my_dataset_train"
	valDatasetName = "my_dataset_val"
	testDatasetName = "my_dataset_test"
	register_coco_instances(trainDatasetName, {}, trainAnnotPath, trainDatasetPath)
	# register_coco_instances(valDatasetName, {}, valAnnotPath, valDatasetPath)
	register_coco_instances(testDatasetName, {}, testAnnotPath, testDatasetPath)

	if os.path.exists(f"{OUT_DIR}/cfg.yaml"):
		print("Found existing cfg, loading...")
		cfg = get_cfg()
		cfg.merge_from_file(f"{OUT_DIR}/cfg.yaml")
	else:
		print("Creating new cfg...")
		cfg = InitCfg(trainDatasetName, valDatasetName, trained=False)
		SaveCfg(cfg)

	#VisualizeAnnotations(cfg.DATASETS.TRAIN[0], toSave=True, imgIndex=0)
	#VisualizeAnnotations(testDatasetName, toSave=True, imgIndex=0)

	### validation during training ### NOTE: breaking changes, needs LossEvalHook
	#evaluator = COCOEvaluator(valDatasetName, ("bbox", "segm"), False, output_dir=cfg.OUTPUT_DIR) # just for validation
	#MetadataCatalog.get(trainDatasetName).evaluator_type = evaluator # just for validation

	# cfg.MODEL.WEIGHTS = f"{OUT_DIR}/model_0014999.pth"  # can resume custom pth, but needs RESUME_CHECKPOINT=False
	trainer = LoadTrainer(cfg, resumeTraining=RESUME_CHECKPOINT)

	if DO_TRAIN:
		trainer.train()
		cfg.MODEL.WEIGHTS = f"{OUT_DIR}/model_final.pth"  # needed after training for visualizing predictions
		SaveCfg(cfg)  # update cfg file

	if DO_INFERENCE:
		for file in os.listdir(PRED_INPUT_FOLDER):
			if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".bmp"):
				print(os.path.join(PRED_INPUT_FOLDER, file))
				VisualizePredictions(cfg, toSave=True, thresh=0.5, imgPath=os.path.join(PRED_INPUT_FOLDER, file))

	if DO_TEST:
		evaluator = COCOEvaluator(testDatasetName, ("bbox", "segm"), False, output_dir=cfg.OUTPUT_DIR)
		val_loader = build_detection_test_loader(cfg, testDatasetName)
		evaluation_res = inference_on_dataset(trainer.model, val_loader, evaluator)
		with open(OUT_DIR + '/evaluation_result_tmp.json', 'w') as outfile:
			json.dump(evaluation_res, outfile, indent=2)


if __name__ == "__main__":
	os.makedirs(OUT_DIR, exist_ok=True)
	main()
