import torch
import numpy as np
import random
import os

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint

from model import Detr
from argparse_utils import parse_args


def main(args):
    RANDOM_SEED = args.seed

    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    
    if args.eval:
        model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, train=False)

        trainer = Trainer(accelerator="gpu") # accelerator를 안쓰면 굉장히 느려짐
        ckpt_path = os.path.join(args.best_map_path, os.listdir(args.best_map_path)[0])
        trainer.test(model, ckpt_path=ckpt_path)
        
        return
    
    if args.inference:
        os.makedirs(f'{args.result_directory}/data_csv')
        os.makedirs(f'{args.result_directory}/img')


        model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, train=False)

        trainer = Trainer(accelerator="gpu")
        ckpt_path = os.path.join(args.best_map_path, os.listdir(args.best_map_path)[0])
        trainer.predict(model, ckpt_path=ckpt_path)

        return


    # Train Model with Pytorch Lightning
    model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)

    # Checkpoint (Pytorch Lightning)
    # mAP@50 값이 큰 모델 저장
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="validation_mAP@50",
        mode="max",
        dirpath=args.best_map_path,
        filename="detr-{epoch:03d}-{validation_mAP@50}",
    )

    # Wandb
    wandb_logger = WandbLogger(project=args.project,
                               name=args.name)
                            #    log_model="all") # log_model="all"하면 속도가 느려짐
    
    wandb.watch(model)

    # pytorch_lightning < 2.0.0
    trainer = Trainer(gpus=1,
                      max_epochs=args.max_epochs,
                      gradient_clip_val=0.1,
                      accumulate_grad_batches=8,
                      log_every_n_steps=20,
                      logger=wandb_logger,
                      callbacks=[checkpoint_callback],
                      resume_from_checkpoint=args.resume_path)

    # pytorch_lightning >= 2.0.0
    # trainer = Trainer(devices=1, accelerator="gpu", max_epochs=MAX_EPOCHS, gradient_clip_val=0.1, accumulate_grad_batches=8, log_every_n_steps=5)

    trainer.fit(model)

    # Save Huggingface Model
    model.model.save_pretrained(args.model_path)

    wandb.finish()


if __name__=='__main__':
    args = parse_args()
    main(args)
