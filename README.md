# Modernized Insurance Claims Fraud Detection (Autoencoder · Two-Tower · GNN · XGBoost)
This project modernizes a legacy insurance-claims fraud detection workflow by integrating
Autoencoder anomaly scores, Two-Tower (Claimant × Provider) embeddings, and GNN-derived graph
features into an XGBoost model for calibrated fraud-risk scoring. 

## Run with Docker
```
docker build --no-cache -t fraud-api .
docker run -p 8000:8000 -e PORT=8000 fraud-api
```

## Run locally
```
conda env create -f environment.yml
conda activate fraud-env
uvicorn app:app --reload
```

## Production architecture
```
         HF Spaces (UI)
           ↓
         Render API
           ↓
         Kafka
           ↓
         AWS Lambda
           ↓
        DynamoDB/S3
           ↑
         HF Spaces (UI)
```

### Training order

1. Synthetic data
python -m src.data.generate_synthetic_data

2. Autoencoder (AE score + latent)
python -m src.autoencoder.train

3. Two-tower (contrastive embeddings)
python -m src.twotower.train

4. GraphSAGE GNN (graph embeddings)
python -m src.gnn.train

5. LightGBM integration
python -m src.lightgbm_model.train_lgbm




