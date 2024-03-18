from model import Detr
from pytorch_lightning import Trainer
from transformers import DetrForObjectDetection
from argparse_utils import parse_args


if __name__=="__main__":
    args = parse_args()

    #Loade Model
    model = DetrForObjectDetection.from_pretrained(args.checkpoint)
    model.to(args.device)

    # Train Model with Pytorch Lightning
    model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)

    # pytorch_lightning < 2.0.0
    trainer = Trainer(gpus=1, max_epochs=args.max_epochs, gradient_clip_val=0.1, accumulate_grad_batches=8, log_every_n_steps=5)

    # pytorch_lightning >= 2.0.0
    # trainer = Trainer(devices=1, accelerator="gpu", max_epochs=MAX_EPOCHS, gradient_clip_val=0.1, accumulate_grad_batches=8, log_every_n_steps=5)

    trainer.fit(model)


    # Save Model
    model.model.save_pretrained(args.model_path)