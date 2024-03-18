from dataset import CocoDetection
from torch.utils.data import DataLoader
from collate import collate_fn
from transformers import DetrImageProcessor
from argparse_utils import parse_args


def make_dataset(mode_directory, train=False):
    args = parse_args()
    image_processor = DetrImageProcessor.from_pretrained(args.checkpoint)

    # Data Loader
    DATASET = CocoDetection(
        image_directory_path=mode_directory,
        image_processor=image_processor,
        train=train)
    
    return DATASET


def make_dataloader(mode_directory, batch_size, train=False, shuffle=False):
    DATASET = make_dataset(mode_directory, train)
    DATALOADER = DataLoader(dataset=DATASET, collate_fn=collate_fn, batch_size=batch_size, shuffle=shuffle)

    return DATALOADER
