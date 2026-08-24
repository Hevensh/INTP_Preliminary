"""ImageNet-100 experiments shared by the ViT and INTP variants."""

from .data import ImageFolderSplits, discover_imagefolder_splits

__all__ = ["ImageFolderSplits", "discover_imagefolder_splits"]
