"""Configuration constants for real-world demo pipeline.

Defines domain filters, file patterns, and directory paths used across all modules.
"""

from dataclasses import dataclass
from pathlib import Path

# Base directories
MODULE_DIR = Path(__file__).parent
DATA_DIR = MODULE_DIR / "data"
CLONED_DIR = MODULE_DIR / "cloned_repos"
COLLECTED_DIR = MODULE_DIR / "collected_code"
REPORTS_DIR = MODULE_DIR / "reports"

# Leakage paper data source directories
LEAKAGE_PAPER_DATA_DIR = DATA_DIR / "leakage_paper"
LEAKAGE_PAPER_COLLECTED_DIR = COLLECTED_DIR / "leakage_paper"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
CLONED_DIR.mkdir(exist_ok=True)
COLLECTED_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
LEAKAGE_PAPER_DATA_DIR.mkdir(exist_ok=True)
LEAKAGE_PAPER_COLLECTED_DIR.mkdir(exist_ok=True)

# HuggingFace dataset names
HF_PAPERS_DATASET = "pwc-archive/papers-with-abstracts"
HF_LINKS_DATASET = "pwc-archive/links-between-paper-and-code"

# Domain filtering keywords (match against tasks array)
# Each domain has keywords that appear in PWC task names
SCIENTIFIC_DOMAINS: dict[str, list[str]] = {
    "biology": [
        "protein",
        "gene",
        "cell",
        "dna",
        "rna",
        "genomic",
        "genome",
        "transcriptome",
        "proteome",
        "phylogenetic",
        "enzyme",
        "binding",
        "folding",
        "sequencing",
        "alignment",
    ],
    "chemistry": [
        "molecule",
        "drug",
        "compound",
        "chemical",
        "molecular",
        "reaction",
        "synthesis",
        "retrosynthesis",
        "property-prediction",
        "toxicity",
        "solubility",
    ],
    "medical": [
        "medical",
        "clinical",
        "disease",
        "diagnosis",
        "pathology",
        "radiology",
        "histopathology",
        "ecg",
        "eeg",
        "survival",
        "patient",
        "healthcare",
        "cancer",
        "tumor",
        "lesion",
    ],
    "physics": [
        "physics",
        "quantum",
        "particle",
        "cosmology",
        "astrophysics",
        "gravitational",
        "fluid",
        "dynamics",
        "simulation",
        "pde",
        "differential-equation",
    ],
    "materials": [
        "material",
        "crystal",
        "alloy",
        "polymer",
        "semiconductor",
        "battery",
        "catalysis",
        "surface",
    ],
    "neuroscience": [
        "brain",
        "neural",
        "neuroscience",
        "fmri",
        "neuroimaging",
        "connectome",
        "spike",
        "eeg",
    ],
    "earth_science": [
        "climate",
        "weather",
        "earthquake",
        "seismic",
        "ocean",
        "atmosphere",
        "satellite",
        "remote-sensing",
        "geospatial",
        "crop",
        "agriculture",
    ],
    "astronomy": [
        "galaxy",
        "stellar",
        "exoplanet",
        "supernova",
        "telescope",
        "redshift",
        "cosmic",
    ],
}

# ML venues to exclude (we want science venues, not ML conferences)
# These are primarily ML/AI methodology venues, not applied science
EXCLUDE_VENUES: list[str] = [
    # Top ML conferences
    "neurips",
    "nips",
    "icml",
    "iclr",
    # AI conferences
    "aaai",
    "ijcai",
    "uai",
    "aistats",
    "colt",
    # NLP conferences
    "acl",
    "emnlp",
    "naacl",
    "coling",
    # Computer vision conferences
    "cvpr",
    "iccv",
    "eccv",
    "bmvc",
    # Graphics/multimedia
    "siggraph",
    # Data mining / web
    "kdd",
    "www",
    "wsdm",
    "recsys",
    "cikm",
    # Robotics
    "icra",
    "iros",
    "corl",
    # Systems
    "mlsys",
    "osdi",
    "sosp",
]

# Tasks to exclude (pure ML infrastructure, not applied science)
EXCLUDE_TASK_KEYWORDS: list[str] = [
    "benchmark",
    "image-classification",  # Generic CV
    "object-detection",  # Generic CV
    "semantic-segmentation",  # Generic CV
    "image-generation",  # GenAI
    "text-generation",  # GenAI
    "language-model",  # LLM infra
    "question-answering",  # NLP infra
    "machine-translation",  # NLP infra
    "speech-recognition",  # Generic speech
    "reinforcement-learning",  # RL framework
    "representation-learning",  # ML research
    "self-supervised",  # ML research
    "contrastive-learning",  # ML research
    "adversarial",  # ML research
    "neural-architecture",  # Architecture search
    "pruning",  # Model compression
    "quantization",  # Model compression
    "distillation",  # Model compression
    "diffusion",  # GenAI infra
]

# ML library imports to detect (indicates ML code)
ML_IMPORTS: list[str] = [
    "sklearn",
    "scikit-learn",
    "torch",
    "pytorch",
    "tensorflow",
    "keras",
    "xgboost",
    "lightgbm",
    "catboost",
    "jax",
    "optax",
    "flax",
]

# Scientific computing imports (supporting evidence for ML code)
SCIENTIFIC_IMPORTS: list[str] = [
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "seaborn",
    "statsmodels",
    "biopython",
    "rdkit",
    "pymatgen",
    "ase",
    "mdtraj",
    "nibabel",
    "nilearn",
    "mne",
]

# Files to exclude from collection
EXCLUDE_FILES: list[str] = [
    "setup.py",
    "setup.cfg",
    "conftest.py",
    "__init__.py",
    "_version.py",
    "version.py",
    "conf.py",  # Sphinx config
    "manage.py",  # Django
]

# File patterns to exclude
EXCLUDE_PATTERNS: list[str] = [
    "test_*.py",
    "*_test.py",
    "tests/*.py",
    "test/*.py",
    "docs/*.py",
    "examples/*.py",  # Often simplified demos
    "benchmarks/*.py",
]

# Clone settings
DEFAULT_CLONE_TIMEOUT = 120  # seconds
DEFAULT_MAX_CONCURRENT_CLONES = 3

# vLLM settings
DEFAULT_LLM_URL = "http://localhost:5001"
DEFAULT_PREFILTER_CONCURRENT = 10
DEFAULT_PREFILTER_TIMEOUT = 30

# File filtering settings
MIN_FILE_SIZE_BYTES = 500  # Skip very small files
MAX_FILE_SIZE_BYTES = 500_000  # Skip very large files (500KB)
MIN_LINES = 20  # Skip trivial files

# Abstract filter settings (LLM-based filtering)
# High concurrency OK - abstracts are small, maximize KV cache utilization
DEFAULT_ABSTRACT_FILTER_CONCURRENT = 100
DEFAULT_ABSTRACT_FILTER_TIMEOUT = 15

# Analysis concurrency settings (by data source)
# Papers with code: high concurrency - large batches benefit from parallelism
# Leakage paper: lower concurrency - notebooks are larger, avoid overload
ANALYSIS_CONCURRENCY: dict[str, int] = {
    "papers_with_code": 60,  # Capped to stay out of KV-cache preemption zone
    "real_world_demo": 60,  # Same as papers_with_code (default source)
    "leakage_paper": 50,  # Lower concurrency for larger notebooks
}
DEFAULT_ANALYSIS_CONCURRENT = 60  # Default if source not in dict


# Data source configurations for run_analysis.py --source shortcut
# Maps source name -> (manifest_path, base_dir, default_patterns)
@dataclass
class DataSourceConfig:
    """Configuration for a data source."""

    manifest: Path
    base_dir: Path
    default_patterns: list[str] | None = None


DATA_SOURCE_CONFIGS: dict[str, DataSourceConfig] = {
    "leakage_paper": DataSourceConfig(
        manifest=LEAKAGE_PAPER_DATA_DIR / "manifest.csv",
        base_dir=LEAKAGE_PAPER_COLLECTED_DIR,
        default_patterns=["ml-001", "ml-007", "ml-009", "ml-010"],
    ),
    "papers_with_code": DataSourceConfig(
        manifest=COLLECTED_DIR / "manifest.csv",
        base_dir=COLLECTED_DIR,
        default_patterns=None,  # Run all patterns
    ),
}
