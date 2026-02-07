"""
Configuration settings for JournalSense Backend
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "faiss_indices"

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

# API Settings
OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL", "research@journalsense.ai")
OPENALEX_BASE_URL = "https://api.openalex.org"

# Model Settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SPACY_MODEL = "en_core_web_sm"

# FAISS Settings
FAISS_DIMENSION = 384  # Matches MiniLM embedding dimension
FAISS_NLIST = 100  # Number of clusters for IVF index

# PDF Processing
MAX_PDF_SIZE_MB = 50
ALLOWED_EXTENSIONS = {'.pdf'}

# Keyword Canonicalization Mappings
CANONICAL_ARCHITECTURES = {
    "vision transformer": "vision_transformer",
    "vit": "vision_transformer",
    "transformer-based encoder": "vision_transformer",
    "swin transformer": "swin_transformer",
    "swin": "swin_transformer",
    "unet": "unet",
    "u-net": "unet",
    "resnet": "resnet",
    "residual network": "resnet",
    "bert": "bert",
    "gpt": "gpt",
    "diffusion model": "diffusion_model",
    "gan": "gan",
    "generative adversarial": "gan",
    "cnn": "cnn",
    "convolutional neural network": "cnn",
    "lstm": "lstm",
    "long short-term memory": "lstm",
    "attention mechanism": "attention",
    "self-attention": "self_attention",
    "cross-attention": "cross_attention",
}

CANONICAL_DATASETS = {
    "brats": "brats",
    "brats2020": "brats",
    "brats2021": "brats",
    "isic": "isic",
    "isic2018": "isic",
    "isic2019": "isic",
    "imagenet": "imagenet",
    "coco": "coco",
    "ms coco": "coco",
    "cifar": "cifar",
    "cifar-10": "cifar10",
    "cifar-100": "cifar100",
    "mnist": "mnist",
    "pascal voc": "pascal_voc",
    "ade20k": "ade20k",
    "cityscapes": "cityscapes",
    "kitti": "kitti",
}

CANONICAL_METRICS = {
    "dice": "dice_score",
    "dice score": "dice_score",
    "dice coefficient": "dice_score",
    "iou": "iou",
    "intersection over union": "iou",
    "jaccard": "iou",
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1_score",
    "f1-score": "f1_score",
    "auc": "auc",
    "roc-auc": "auc",
    "map": "map",
    "mean average precision": "map",
    "psnr": "psnr",
    "ssim": "ssim",
    "bleu": "bleu",
    "rouge": "rouge",
}

CANONICAL_TASKS = {
    "segmentation": "segmentation",
    "image segmentation": "segmentation",
    "semantic segmentation": "semantic_segmentation",
    "instance segmentation": "instance_segmentation",
    "classification": "classification",
    "image classification": "image_classification",
    "object detection": "object_detection",
    "detection": "object_detection",
    "generation": "generation",
    "image generation": "image_generation",
    "text generation": "text_generation",
    "translation": "translation",
    "machine translation": "translation",
    "summarization": "summarization",
    "question answering": "question_answering",
    "qa": "question_answering",
    "ner": "named_entity_recognition",
    "named entity recognition": "named_entity_recognition",
}

# Custom spaCy entity labels for research papers
RESEARCH_ENTITY_LABELS = [
    "MODEL",      # Architecture names: ViT, UNet, BERT
    "DATASET",    # Dataset names: BraTS, ISIC, ImageNet
    "METRIC",     # Evaluation metrics: Dice, IoU, F1
    "TASK",       # Research tasks: segmentation, classification
    "BASELINE",   # Baseline models compared against
    "LIMITATION", # Stated limitations
    "CONTRIBUTION", # Key contributions
    "METHOD",     # Methodology components
]
