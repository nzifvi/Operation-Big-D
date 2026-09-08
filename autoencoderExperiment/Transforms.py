import torch
import cv2
import numpy

"""
To implement:
1) Implement 3 stage transformation (1 -> 2 -> 3)
"""

def apply1StageTransformation(tensor : torch.Tensor) -> torch.Tensor:
    """
    Normalises input tensor to a real number domain of [0, 1].
    """
    return tensor / 255.0

def apply2StageTransformation(tensor : torch.Tensor) -> torch.Tensor:
    """
    Applies 2 transformation stages. Firstly, function normalises to a real number domain of [0 ,1]. Secondly, function
    applies a Canny filter.
    """
    edges = cv2.Canny(
        tensor.squeeze().numpy().astype(numpy.uint8),
        100,
        200
    )
    return apply1StageTransformation(
        torch.from_numpy(edges).unsqueeze(0).float()
    )

def apply3StageTransformation(tensor : torch.Tensor) -> torch.Tensor:
    ...

