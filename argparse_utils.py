import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='DETR Training Settings')

    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda:0 for GPU or cpu for CPU)')
    parser.add_argument('--checkpoint', type=str, default='facebook/detr-resnet-50',
                        choices=['facebook/detr-resnet-50', 'SenseTime/deformable-detr'], help='DETR checkpoint model')
    parser.add_argument('--confidence-threshold', type=float, default=0.5, help='Confidence threshold for predictions')
    parser.add_argument('--iou-threshold', type=float, default=0.5, help='IOU threshold for predictions')
    parser.add_argument('--annotation-file-name', type=str, default='_annotations.coco.json', help='Annotation file name')
    parser.add_argument('--train-directory', type=str, default='/home/path/train', help='Directory containing training data')
    parser.add_argument('--val-directory', type=str, default='/home/path/valid', help='Directory containing validation data')
    parser.add_argument('--test-directory', type=str, default='/home/path/test', help='Directory containing test data (labeled)')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--max-epochs', type=int, default=500, help='Maximum number of epochs to train')
    parser.add_argument('--model-path', type=str, default='/home/detr_exp', help='Path to save the trained model')
    parser.add_argument('--resume-path', type=str, default=None, help='Chechpoint model path for resume training')
    parser.add_argument('--best-map-path', type=str, default='/home/detr_best_checkpoint', help='Save best mAp@50 model here')
    parser.add_argument('--project', type=str, default='hanyang', help='Wandb project name')
    parser.add_argument('--name', type=str, default='DETR', help='Wandb dashboard name')
    parser.add_argument('--infer-directory', type=str, default='/home/data', help='Directory containing test data(unlabeled)')
    parser.add_argument('--result-directory', type=str, default='/home/detr_result', help='Directory of inference results')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--seed', type=int, default=42)


    parser.add_argument('--num_iters', type=int, default=40, help='total iters to benchmark speed')
    parser.add_argument('--warm_iters', type=int, default=5, help='ignore first several iters that are very slow')
    parser.add_argument('--batch_size_test', type=int, default=1, help='batch size in inference')

    args = parser.parse_args()
    return args
