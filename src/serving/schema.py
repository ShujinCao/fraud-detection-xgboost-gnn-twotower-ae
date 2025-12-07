from pydantic import BaseModel

class ClaimFeatures(BaseModel):
    claim_id: int
    claimant_id: int
    provider_id: int
    claim_amount: float
    procedure_code: int
    days_since_last_claim: int

