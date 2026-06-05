# openmlkit/__init__.py

from .labelmap import LabelMap
from .detector import TextDetector
from .recognizer import TextRecognizer
from .pipeline import OpenMLKitOCR

__version__ = "1.0.3"
__all__ = ["OpenMLKitOCR", "TextDetector", "TextRecognizer", "LabelMap"]
