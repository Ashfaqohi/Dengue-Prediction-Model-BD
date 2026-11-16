# Dengue prediction model

This project trains a scikit-learn dengue classification model, saves the complete preprocessing/model pipeline with `joblib`, and exposes predictions through FastAPI.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Train the final model

From the repository root:

```bash
python train.py
```

This cleans `Data/dataset.csv`, writes `Data/dataset_cleaned.csv`, and saves the selected model to `Models/best_dengue_model.joblib` (metrics go to `Models/best_dengue_model.metrics.json`). The model file contains preprocessing, so the API accepts the original feature values.

## Run the API

```bash
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API. Use `GET /health` for a health check and `POST /predict` with a JSON object containing the training feature names under `features`.

## Build and run the container

```bash
docker build -t dengue-api:latest .
docker run --rm -p 8000:8000 dengue-api:latest
```

Or use Docker Compose:

```bash
docker compose up --build
```

The test URL is `http://127.0.0.1:8000/docs` and the health URL is `http://127.0.0.1:8000/health`.

## Deploy to Kubernetes

The manifests in `k8s/` work with minikube, kind, or a remote cluster. For a local minikube cluster:

```bash
eval "$(minikube docker-env)"
docker build -t dengue-api:latest .
./deploy.sh
kubectl -n dengue port-forward service/dengue-api 8000:80
```

For a remote cluster, push the image to a registry, then deploy it by setting `IMAGE`:

```bash
docker tag dengue-api:latest REGISTRY/PROJECT/dengue-api:latest
docker push REGISTRY/PROJECT/dengue-api:latest
IMAGE=REGISTRY/PROJECT/dengue-api:latest ./deploy.sh
```

After port-forwarding, test at `http://127.0.0.1:8000/docs`. The deployment has readiness and liveness checks on `/health`.

## Re-run the notebook

Activate the virtual environment first, then run from the `Notebook` directory so its existing relative data paths resolve:

```bash
cd Notebook
jupyter nbconvert --to notebook --execute Dengue_bd.ipynb --output Dengue_bd.executed.ipynb
```

The notebook performs the same cleaning and model-selection workflow and can be opened normally with `jupyter notebook Dengue_bd.ipynb`.
