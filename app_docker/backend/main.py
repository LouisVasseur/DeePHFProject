"""
DeePHF Web Interface — FastAPI Backend
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import os
import time

from inference import DeePHFInference, parse_xyz, smiles_to_xyz

app = FastAPI(title="DeePHFProject — Correlation Energy Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="/app/frontend"), name="static")

engine: DeePHFInference = None


@app.on_event("startup")
async def startup():
    global engine
    engine = DeePHFInference(
        model_elec_path = os.environ.get("MODEL_ELEC", "/app/model_elec.pt"),
        model_soap_path = os.environ.get("MODEL_SOAP", "/app/model_soap.pt"),
        model_comb_path = os.environ.get("MODEL_COMB", "/app/model_comb.pt"),
        cache_dir       = os.environ.get("CACHE_DIR",  "/app/cache_electronic"),
        hf_basis        = os.environ.get("HF_BASIS",   "cc-pvdz"),
        w_atomic        = float(os.environ.get("W_ATOMIC", "1.0")),
    )
    loaded = [k for k, m in [("electronic", engine.model_elec), ("SOAP", engine.model_soap), ("combined", engine.model_comb)] if m is not None]
    print(f"[DeePHFProject] Backend ready. Models loaded: {loaded if loaded else 'none (demo mode)'}")


@app.get("/")
async def root():
    return FileResponse("/app/frontend/index.html")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models": {
            "electronic": engine.model_elec is not None,
            "soap":       engine.model_soap is not None,
            "combined":   engine.model_comb is not None,
        }
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(".xyz"):
        raise HTTPException(status_code=400, detail="Only .xyz files are supported.")

    content = await file.read()
    try:
        xyz_text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File is not valid UTF-8 text.")

    try:
        atomic_numbers, coords, n_atoms, formula = parse_xyz(xyz_text)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse .xyz file: {str(e)}")

    try:
        t0 = time.time()
        result = engine.predict(atomic_numbers, coords)
        result["elapsed_s"] = round(time.time() - t0, 2)
        result["xyz"] = xyz_text
        result["n_atoms"]   = n_atoms
        result["formula"]   = formula
        result["filename"]  = file.filename
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    

class SmilesRequest(BaseModel):
    smiles: str

@app.post("/predict_smiles")
async def predict_smiles(req: SmilesRequest):
    smiles = req.smiles.strip()
    if not smiles:
        raise HTTPException(status_code=400, detail="SMILES string is empty.")

    # Step 1 — SMILES → XYZ text (replaces the file upload + decode steps above)
    try:
        xyz_text = smiles_to_xyz(smiles)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Step 2 — parse XYZ (identical to /predict from here on)
    try:
        atomic_numbers, coords, n_atoms, formula = parse_xyz(xyz_text)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse generated XYZ: {str(e)}")

    # Step 3 — run the prediction engine (identical to /predict)
    try:
        t0 = time.time()
        result = engine.predict(atomic_numbers, coords)
        result["elapsed_s"] = round(time.time() - t0, 2)
        result["xyz"] = xyz_text
        result["n_atoms"]   = n_atoms
        result["formula"]   = formula
        result["filename"]  = smiles   # no file, so we use the SMILES as the identifier
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)
