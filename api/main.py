"""HTTP API with explicit request validation."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.predict import predict_next_24

app = FastAPI(title='Germany Electricity Load Forecast')


class PredictionRequest(BaseModel):
    data_path: str = Field(default='data/sample_data.csv', min_length=1)


@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.post('/predict')
def predict(request: PredictionRequest) -> dict:
    try:
        return {'forecast': predict_next_24(request.data_path)}
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
