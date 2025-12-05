from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

N_CLAIMANTS = 10_000
N_PROVIDERS = 1_500
N_CLAIMS = 50_000

RANDOM_SEED = 42

# Model dims
AE_INPUT_DIM = 6      # number of numeric features we feed into AE
AE_LATENT_DIM = 8

EMBED_DIM = 16        # for two-tower & GNN outputs

