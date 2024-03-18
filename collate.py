from transformers import DetrImageProcessor
from argparse_utils import parse_args


def collate_fn(batch):
    args = parse_args()

    # image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)
    image_processor = DetrImageProcessor.from_pretrained(args.checkpoint)

    # DETR authors employ various image sizes during training, making it not possible
    # to directly batch together images. Hence they pad the images to the biggest
    # resolution in a given batch, and create a corresponding binary pixel_mask
    # which indicates which pixels are real/which are padding
    pixel_values = [item[0] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }