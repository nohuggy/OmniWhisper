import warnings
from importlib.metadata import PackageNotFoundError, version

warnings.filterwarnings("ignore", module="torchaudio")
warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    message="invalid escape sequence",
    module="pydub.utils",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="torch.distributed.algorithms.ddp_comm_hooks",
)

try:
    __version__ = version("omnivoice")
except PackageNotFoundError:
    __version__ = "0.0.0"

import os

# Automatically switch to Kaggle-optimized model if in Kaggle environment
if os.environ.get("KAGGLE_URL_BASE") or os.path.exists("/kaggle"):
    from omnivoice.models.omnivoice_kaggle import (
        OmniVoice,
        OmniVoiceConfig,
        OmniVoiceGenerationConfig,
    )
else:
    from omnivoice.models.omnivoice import (
        OmniVoice,
        OmniVoiceConfig,
        OmniVoiceGenerationConfig,
    )

__all__ = ["OmniVoice", "OmniVoiceConfig", "OmniVoiceGenerationConfig"]
