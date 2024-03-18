import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='DETR Training Settings')

    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda:0 for GPU or cpu for CPU)')
    parser.add_argument('--checkpoint', type=str, default='facebook/detr-resnet-50', help='DETR checkpoint model')
    parser.add_argument('--confidence-threshold', type=float, default=0.5, help='Confidence threshold for predictions')
    parser.add_argument('--iou-threshold', type=float, default=0.5, help='IOU threshold for predictions')
    parser.add_argument('--annotation-file-name', type=str, default='_annotations.coco.json', help='Annotation file name')
    parser.add_argument('--train-directory', type=str, default='/home/path/train', help='Directory containing training data')
    parser.add_argument('--val-directory', type=str, default='/home/path/valid', help='Directory containing validation data')
    parser.add_argument('--test-directory', type=str, default='/home/path/test', help='Directory containing test data (labeled)')
    parser.add_argument('--max-epochs', type=int, default=300, help='Maximum number of epochs to train')
    parser.add_argument('--model-path', type=str, default='/home/custom-model', help='Path to save the trained model')
    parser.add_argument('--infer-directory', type=str, default='/home/data', help='Directory containing test data(unlabeled)')
    parser.add_argument('--result-directory', type=str, default='/home/result', help='Directory of inference results')

    args = parser.parse_args()
    return args