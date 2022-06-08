from typing import Tuple, List, Dict, Any, Optional, Callable
from itertools import repeat
from collections import defaultdict
from torchvision.datasets.coco import CocoDetection


class CocoDetectionMaskRCNN(CocoDetection):
    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, annFile, transform, self.target_transform, transforms)

    def transform_target(
        self, targets: Tuple[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Transform the target to the format required by the model.
        """
        batch_size = len(targets)
        transformed_target = [defaultdict(list) for _ in repeat(None, batch_size)]
        for t in targets:
            for ann in t:
                transformed_target["boxes"].append(self.xywh2xyxy(ann["bbox"]))
                transformed_target["labels"].append(ann["category_id"])
                transformed_target["masks"].append(
                    self.coco.annToMask(ann["segmentation"])
                )

        return transformed_target

    @staticmethod
    def xywh2xyxy(x: List[int, int, int, int]) -> List[int, int, int, int]:
        """
        Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2].
        """
        return [x[0], x[1], x[0] + x[2], x[1] + x[3]]
