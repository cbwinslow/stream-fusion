"""StreamFusion vision processing and keyframe optimization package."""

from stream_fusion.vision.dense_extractor import DenseFrameExtractor
from stream_fusion.vision.optimizer import AdaptiveFrameOptimizer
from stream_fusion.vision.processor import VisionProcessor

__all__ = ["VisionProcessor", "DenseFrameExtractor", "AdaptiveFrameOptimizer"]
