import cv2
import torch
import supervision as sv
import os
from transformers import DetrForObjectDetection, DetrImageProcessor
import glob
import csv
from argparse_utils import parse_args


def create_new_directory(path):
    original_path = path
    counter = 1
    while os.path.exists(path):
        path = f"{original_path}{counter}"
        counter += 1
    os.mkdir(path)
    return path


if __name__=='__main__':
    args = parse_args()

    model = DetrForObjectDetection.from_pretrained(args.model_path)
    model.to(args.device)

    image_processor = DetrImageProcessor.from_pretrained(args.checkpoint)

    infer_directory = args.infer_directory
    image_lst = glob.glob(f"{infer_directory}/*.png")

    result_path = create_new_directory(args.result_directory)

    csv_path = create_new_directory(os.path.join(result_path, 'data_csv'))
    results_path = create_new_directory(os.path.join(result_path, 'img'))

    time_li = []
    for i in range(len(image_lst)):
        image = cv2.imread(image_lst[i])
        image_id = image_lst[i].split('/')[-1][:-4]
        box_annotator = sv.BoxAnnotator()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        with torch.no_grad():

            # load image and predict
            inputs = image_processor(images=image, return_tensors='pt').to(args.device)
            start_event.record()
            outputs = model(**inputs)
            end_event.record()

            # post-process
            target_sizes = torch.tensor([image.shape[:2]]).to(args.device)
            results = image_processor.post_process_object_detection(
                outputs=outputs, 
                threshold=args.confidence_threshold, 
                target_sizes=target_sizes
            )[0]

        torch.cuda.synchronize()
        time_taken = start_event.elapsed_time(end_event)
        time_li.append(time_taken * 1e-3)

        detections = sv.Detections.from_transformers(transformers_results=results).with_nms(threshold=args.iou_threshold)
        class_id = ["cluster" for _ in detections.class_id]
        confidence = [f"{confidence:.2f}" for confidence in detections.confidence]
        labels = [f"{id} {score}" for id, score in zip(class_id, confidence)]
        info = []
        for j in range(len(detections)):
            frame = box_annotator.annotate(scene=image, detections=detections[j], labels=[labels[j]])
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
        with open(f'{csv_path}/{image_id}.csv', 'w', newline='') as csvfile:
            # CSV writer 객체 생성
            writer = csv.writer(csvfile)
            
            # 첫 번째 행에 열 이름 추가
            writer.writerow(columns)
            
            # 데이터를 반복하여 파일에 추가
            for data in info:  # 여기서 your_data는 데이터를 담고 있는 리스트나 반복 가능한 객체여야 합니다.
                # 데이터를 쉼표로 구분하여 파일에 추가
                writer.writerow(data)


        pred_img_path = os.path.join(results_path, f"pred_img_{image_id}.png")
        cv2.imwrite(pred_img_path, image)

    print(f"Average time on GPU pre image: {sum(time_li)/len(img_lst)}")
