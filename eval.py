import torch
from coco_eval import CocoEvaluator
from tqdm import tqdm
from data import make_dataset, make_dataloader
from test_fn import prepare_for_coco_detection
from transformers import DetrForObjectDetection, DetrImageProcessor
from argparse_utils import parse_args
from get_iou import get_iou


if __name__=='__main__':
    args = parse_args()

    model = DetrForObjectDetection.from_pretrained(args.model_path)
    model.to(args.device)

    image_processor = DetrImageProcessor.from_pretrained(args.checkpoint)

    TEST_DATASET = make_dataset(args.test_directory)
    count = len(TEST_DATASET)

    evaluator = CocoEvaluator(coco_gt=TEST_DATASET.coco, iou_types=["bbox"])

    print("Running evaluation...")

    TEST_DATALOADER = make_dataloader(args.test_directory, 1)

    miou = 0
    for idx, batch in enumerate(tqdm(TEST_DATALOADER)):
        pixel_values = batch["pixel_values"].to(args.device)
        pixel_mask = batch["pixel_mask"].to(args.device)
        labels = [{k: v.to(args.device) for k, v in t.items()} for t in batch["labels"]]

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
        # 만약 threshold 값을 넘어가는 box가 없으면 results는 빈 리스트가 되고 evaluator에서 에러가 뜨니 주의하자.
        results = image_processor.post_process_object_detection(outputs, target_sizes=orig_target_sizes)

        predictions = {target['image_id'].item(): output for target, output in zip(labels, results)}
        predictions = prepare_for_coco_detection(predictions)
        a = evaluator.update(predictions)

        # Compute IoU
        pred_box_li = results[0]['boxes'].cpu().tolist()
        gt_box_li = labels[0]['boxes'].cpu().tolist()
        iou_matrix = get_iou(pred_box_li, gt_box_li)
        total_iou = sum([sum(sublist) for sublist in iou_matrix])
        pred_box_len = len(pred_box_li) # pred box 개수
        _miou = total_iou / pred_box_len
        miou += _miou
        

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

    print("mIoU :", miou/len(TEST_DATALOADER))
