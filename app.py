#!/usr/bin/env python3
# app.py

import os
import zipfile
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from model.utils.inference_api import init_model, predict_from_folder

app = FastAPI()

# 1) Mount thư mục static
app.mount("/static", StaticFiles(directory="static"), name="static")

# 2) GET / → index.html
@app.get("/", include_in_schema=False)
async def index():
    return FileResponse("static/index.html")

# 3) Load model khi startup
@app.on_event("startup")
def startup():
    global MODEL
    MODEL = init_model(
        checkpoint_path="work/checkpoint/GaitSet/GaitSet_CASIA-B_70_False_256_0.2_128_full_30-50000-encoder.ptm",
        pretr_path="/content/dataset/content/GaitSetv1/data_pretr"
    )

class Prediction(BaseModel):
    subject_id: str
    view: str

def _find_sequence_folder(root_dir: str) -> str:
    """
    Duyệt toàn bộ cây thư mục starting tại root_dir,
    trả về thư mục đầu tiên có chứa file ảnh (.png, .jpg).
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if any(fname.lower().endswith((".png", ".jpg", ".jpeg")) for fname in filenames):
            return dirpath
    raise HTTPException(400, f"Không tìm thấy thư mục con chứa ảnh trong ZIP")

# 4) POST /predict trả về JSON
@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...)):
    # kiểm tra đuôi
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(400, "Chỉ upload file .zip silhouette")

    with tempfile.TemporaryDirectory() as tmp:
        # 1) ghi ZIP vào tmp
        zip_path = os.path.join(tmp, file.filename)
        with open(zip_path, "wb") as f:
            f.write(await file.read())

        # 2) giải nén
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmp)

        # 3) tìm folder con chứa ảnh
        seq_folder = _find_sequence_folder(tmp)

        # 4) predict
        sid, view = predict_from_folder(MODEL, seq_folder)

    # trả về zero-padded view
    return Prediction(subject_id=str(sid), view=str(view).zfill(3))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
