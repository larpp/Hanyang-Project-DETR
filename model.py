import pytorch_lightning as pl
import torch
import os
from coco_eval import CocoEvaluator
import csv
import supervision as sv
import cv2
import json
from argparse_utils import parse_args
from data import make_dataloader, make_dataset
from test_fn import prepare_for_coco_detection


def id2label(directory, train=True):
    # we will use id2label function for training
    TRAIN_DATASET = make_dataset(directory, train)
    categories = TRAIN_DATASET.coco.cats
    return {k: v['name'] for k,v in categories.items()}


def create_new_directory(path):
    original_path = path
    counter = 1
    while os.path.exists(path):
        path = f"{original_path}{counter}"
        counter += 1
    os.mkdir(path)
    return path


class Detr(pl.LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay, train=True):
        super().__init__()
        args = parse_args()
        if args.checkpoint == 'facebook/detr-resnet-50':
            from transformers import DetrForObjectDetection, DetrImageProcessor
            self.model = DetrForObjectDetection.from_pretrained(
                pretrained_model_name_or_path=args.checkpoint,
                num_labels=len(id2label(args.train_directory)),
                ignore_mismatched_sizes=True
            )
            self.image_processor = DetrImageProcessor.from_pretrained(args.checkpoint)
        else:
            from transformers import DeformableDetrForObjectDetection, AutoImageProcessor
            self.model = DeformableDetrForObjectDetection.from_pretrained(
                pretrained_model_name_or_path=args.checkpoint,
                num_labels=len(id2label(args.train_directory)),
                ignore_mismatched_sizes=True
            )
            self.image_processor = AutoImageProcessor.from_pretrained(args.checkpoint)

        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

        if train:
            self.evaluator_train = CocoEvaluator(coco_gt=make_dataset(args.train_directory).coco, iou_types=["bbox"])
            self.evaluator_val = CocoEvaluator(coco_gt=make_dataset(args.val_directory).coco, iou_types=["bbox"])
        elif args.eval:
            self.evaluator_test = CocoEvaluator(coco_gt=make_dataset(args.test_directory).coco, iou_types=["bbox"])

    
    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    
    def common_step(self, batch, batch_idx):
        args = parse_args()

        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
        results = self.image_processor.post_process_object_detection(outputs, target_sizes=orig_target_sizes, threshold=args.confidence_threshold)
   
        predictions = {target['image_id'].item(): output for target, output in zip(labels, results)}
        predictions = prepare_for_coco_detection(predictions)

        return loss, loss_dict, predictions

    
    def training_step(self, batch, batch_idx):
        loss, loss_dict, predictions = self.common_step(batch, batch_idx)
        # logs metrics for each training_step, and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item())

        # predicted 박스가 없을 경우 에러 방지
        try:
            self.evaluator_train.update(predictions)
        except:
            pass

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict, predictions = self.common_step(batch, batch_idx)
        self.log("validation/loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())

        try:
            self.evaluator_val.update(predictions)
        except:
            pass

        return loss

    
    def test_step(self, batch, batch_idx):
        loss, loss_dict, predictions = self.common_step(batch, batch_idx)
        self.log("test/loss", loss)
        for k, v in loss_dict.items():
            self.log("test_" + k, v.item())

        self.evaluator_test.update(predictions)

        return loss
    

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        args = parse_args()
        with open(f'{args.infer_directory}/{args.annotation_file_name}', 'r') as f:
            image = json.load(f)

        box_annotator = sv.BoxAnnotator()

        # result_path = create_new_directory(args.result_directory)

        # if not os.path.exists(f'{result_path}/data_csv'):
        #     create_new_directory(os.path.join(result_path, 'data_csv'))
        # if not os.path.exists(f'{result_path}/img'):
        #     create_new_directory(os.path.join(result_path, 'img'))

        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        # post-process
        target_sizes = labels[0]['orig_size'].unsqueeze(dim=0)
        results = self.image_processor.post_process_object_detection(
                outputs=outputs, 
                threshold=args.confidence_threshold, 
                target_sizes=target_sizes
            )[0]

        image_id = image['images'][int(labels[0]['image_id'].cpu().tolist()[0])]['file_name']

        detections = sv.Detections.from_transformers(transformers_results=results).with_nms(threshold=args.confidence_threshold)
        if len(detections) == 0:
            return
        class_id = ["cluster" for _ in detections.class_id]
        confidence = [f"{confidence:.2f}" for confidence in detections.confidence]
        labels = [f"{id} {score}" for id, score in zip(class_id, confidence)]
        info = []
        scene = cv2.imread(f'{args.infer_directory}/{image_id}')
        for j in range(len(detections)):
            frame = box_annotator.annotate(scene=scene, detections=detections[j], labels=[labels[j]])
            # (x1, y1), (x2, y2)는 각각 bbox의 좌측 윗 부분 좌표와, 우측 하단 부분의 좌표
            # xyxy = array([[x1, y1, x2, y2]]) shape
            x1, y1, x2, y2 = detections[j].xyxy[0]
            w, h = x2 - x1, y2 - y1
            confidence = detections[j].confidence[0]
            area = w * h
            x_c, y_c = x1 + w/2, y1 + h/2

            data = [x_c, y_c, w, h, confidence, area]
            info.append(data)
        
        # 열 이름 정하기
        columns = ['x_center', 'y_center', 'width', 'height', 'confidence_score', 'area']
        # CSV 파일 열기 (이미 존재하면 'a'로 열어서 추가, 존재하지 않으면 'w'로 새로 생성)
        image_id = image_id[:-4]
        with open(f'{args.result_directory}/data_csv/{image_id}.csv', 'w', newline='') as csvfile:
            # CSV writer 객체 생성
            writer = csv.writer(csvfile)
            
            # 첫 번째 행에 열 이름 추가
            writer.writerow(columns)
            
            # 데이터를 반복하여 파일에 추가
            for data in info:  # 여기서 your_data는 데이터를 담고 있는 리스트나 반복 가능한 객체여야 합니다.
                # 데이터를 쉼표로 구분하여 파일에 추가
                writer.writerow(data)


        pred_img_path = os.path.join(f'{args.result_directory}/img', f"{image_id}.png")
        cv2.imwrite(pred_img_path, scene)
    

    def on_train_epoch_end(self):
        args = parse_args()
        try: # model이 predict한 결과가 없을 경우를 대비
            self.evaluator_train.synchronize_between_processes()
            self.evaluator_train.accumulate()
            self.evaluator_train.summarize()
            self.log("training_mAP@50", self.evaluator_train.coco_eval['bbox'].stats[1])
            self.evaluator_train = CocoEvaluator(coco_gt=make_dataset(args.train_directory).coco, iou_types=["bbox"])
        except:
            self.log("training_mAP@50", 0)
            self.evaluator_train = CocoEvaluator(coco_gt=make_dataset(args.train_directory).coco, iou_types=["bbox"])


    def on_validation_epoch_end(self):
        args = parse_args()
        try:
            self.evaluator_val.synchronize_between_processes()
            self.evaluator_val.accumulate()
            self.evaluator_val.summarize()
            self.log("validation_mAP@50", self.evaluator_val.coco_eval['bbox'].stats[1])
            self.evaluator_val = CocoEvaluator(coco_gt=make_dataset(args.val_directory).coco, iou_types=["bbox"])
        except:
            self.log("validation_mAP@50", 0)
            self.evaluator_val = CocoEvaluator(coco_gt=make_dataset(args.val_directory).coco, iou_types=["bbox"])


    def on_test_epoch_end(self):
        self.evaluator_test.synchronize_between_processes()
        self.evaluator_test.accumulate()
        self.evaluator_test.summarize()
        self.log("test_mAP@50", self.evaluator_test.coco_eval['bbox'].stats[1])


    def configure_optimizers(self):
        # DETR authors decided to use different learning rate for backbone
        # you can learn more about it here:
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L22-L23
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L131-L139
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

    
    def train_dataloader(self):
        args = parse_args()
        TRAIN_DATALOADER = make_dataloader(args.train_directory, args.batch_size, True, True)
        return TRAIN_DATALOADER

    
    def val_dataloader(self):
        args = parse_args()
        VAL_DATALOADER =  make_dataloader(args.val_directory, args.batch_size)
        return VAL_DATALOADER

    
    def test_dataloader(self):
        args = parse_args()
        TEST_DATALOADER =  make_dataloader(args.test_directory, args.batch_size)
        return TEST_DATALOADER


    # Inference 할 이미지는 annotations가 없으므로 빈 annoatation.json file을 먼저 만들어주자.
    # predict_ann.py를 먼저 실행
    def predict_dataloader(self):
        args = parse_args()
        PREDICT_DATALOADER = make_dataloader(args.infer_directory, 1)
        return PREDICT_DATALOADER
