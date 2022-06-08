from pathlib import Path
import torch
import transforms as T
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from train_engine import train_one_epoch, evaluate
from coco_utils import get_coco
import train_utils as utils


class Train_MaskRCNN:
    def __init__(self):
        self.coco_root = Path(__file__).parent.absolute().parent.parent / "data/coco"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def create_model(self):
        backbone = resnet_fpn_backbone(
            backbone_name="resnet101",
            pretrained=True,
            norm_layer=misc_nn_ops.FrozenBatchNorm2d,
            trainable_layers=3,
        )
        return MaskRCNN(backbone, num_classes=91)

    def _get_transform(self, train: bool):
        transforms = []
        transforms.append(T.ToTensor())
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)

    def _get_dataloader(self):
        trg_dataset = get_coco(
            self.coco_root,
            "train",
            transforms=self._get_transform(train=True),
            mode="instances",
        )
        trg_dataloader = torch.utils.data.DataLoader(
            trg_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=8,
            collate_fn=utils.collate_fn,
            prefetch_factor=1,
        )

        val_dataset = get_coco(
            self.coco_root,
            "val",
            transforms=self._get_transform(train=False),
            mode="instances",
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            collate_fn=utils.collate_fn,
            prefetch_factor=1,
        )

        return trg_dataloader, val_dataloader

    def train_model(self):
        trg_dataloader, val_dataloader = self._get_dataloader()
        model = self.create_model()
        model.to(self.device)
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=0.1
        )

        num_epochs = 10
        for epoch in range(num_epochs):
            train_one_epoch(
                model, optimizer, trg_dataloader, self.device, epoch, print_freq=10
            )
            lr_scheduler.step()
            evaluate(model, val_dataloader, self.device)


if __name__ == "__main__":
    train = Train_MaskRCNN()
    train.create_model()
    train.train_model()
