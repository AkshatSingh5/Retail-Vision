from vision.detection.types import Detection

# Do not import YOLODetector here. It pulls ultralytics/torch, which must stay
# out of the Vercel serverless bundle.

__all__ = ["Detection"]
