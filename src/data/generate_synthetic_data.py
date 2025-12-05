import numpy as np
import pandas as pd
from pathlib import Path

from src.config import (
    DATA_RAW_DIR,
    N_CLAIMANTS,
    N_PROVIDERS,
    N_CLAIMS,
    RANDOM_SEED,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

def generate_claimants():
    np.random.seed(RANDOM_SEED)
    claimant_ids = np.arange(1, N_CLAIMANTS + 1)
    ages = np.random.randint(18, 90, size=N_CLAIMANTS)
    tenure_months = np.random.randint(1, 120, size=N_CLAIMANTS)
    risk_score = np.random.beta(2, 5, size=N_CLAIMANTS)  # base risk

    df = pd.DataFrame({
        "claimant_id": claimant_ids,
        "claimant_age": ages,
        "claimant_tenure_months": tenure_months,
        "claimant_base_risk": risk_score,
    })
    return df

def generate_providers():
    np.random.seed(RANDOM_SEED + 1)
    provider_ids = np.arange(1, N_PROVIDERS + 1)
    specialties = np.random.randint(0, 10, size=N_PROVIDERS)
    years_in_network = np.random.randint(1, 40, size=N_PROVIDERS)
    provider_risk = np.random.beta(1.5, 8, size=N_PROVIDERS)

    df = pd.DataFrame({
        "provider_id": provider_ids,
        "provider_specialty": specialties,
        "provider_years_in_network": years_in_network,
        "provider_base_risk": provider_risk,
    })
    return df

def generate_claims(claimants: pd.DataFrame, providers: pd.DataFrame):
    np.random.seed(RANDOM_SEED + 2)
    claim_ids = np.arange(1, N_CLAIMS + 1)

    claimant_ids = np.random.choice(claimants["claimant_id"], size=N_CLAIMS)
    provider_ids = np.random.choice(providers["provider_id"], size=N_CLAIMS)

    # Basic per-claim features
    claim_amount = np.random.gamma(shape=2.0, scale=500.0, size=N_CLAIMS)
    procedure_code = np.random.randint(0, 20, size=N_CLAIMS)
    days_since_last_claim = np.random.exponential(scale=30, size=N_CLAIMS).astype(int)
    days_since_last_claim = np.clip(days_since_last_claim, 0, 365)

    df = pd.DataFrame({
        "claim_id": claim_ids,
        "claimant_id": claimant_ids,
        "provider_id": provider_ids,
        "claim_amount": claim_amount,
        "procedure_code": procedure_code,
        "days_since_last_claim": days_since_last_claim,
    })

    # merge risk signals from entities
    df = df.merge(
        claimants[["claimant_id", "claimant_base_risk"]],
        on="claimant_id",
        how="left",
    ).merge(
        providers[["provider_id", "provider_base_risk"]],
        on="provider_id",
        how="left",
    )

    # create a synthetic fraud label with some structure
    base_prob = 0.02  # ~2% baseline fraud
    prob = (
        base_prob
        + 0.1 * df["claimant_base_risk"]
        + 0.1 * df["provider_base_risk"]
        + 0.00002 * df["claim_amount"]
        + 0.01 * (df["days_since_last_claim"] < 5).astype(float)
    )
    prob = np.clip(prob, 0, 0.8)
    df["is_fraud"] = np.random.binomial(1, prob)

    return df

def main():
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Generating synthetic claimants...")
    claimants = generate_claimants()
    logger.info("Generating synthetic providers...")
    providers = generate_providers()
    logger.info("Generating synthetic claims...")
    claims = generate_claims(claimants, providers)

    claimants.to_csv(DATA_RAW_DIR / "claimants.csv", index=False)
    providers.to_csv(DATA_RAW_DIR / "providers.csv", index=False)
    claims.to_csv(DATA_RAW_DIR / "claims.csv", index=False)
    logger.info("Synthetic data written to data/raw")

if __name__ == "__main__":
    main()

