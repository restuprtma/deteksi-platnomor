from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, r"e:\PCVK x ML")

from main_compare import SVMModel, CNNModel, split_characters, correct_plate_context

app = FastAPI(title="Plate Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

svm_model = None
cnn_model = None

DATASET_PATH = r"e:\PCVK x ML\caasperart\haarcascadeplatenumber\versions\10\DatasetCharacter\DatasetCharacter"

@app.on_event("startup")
async def load_models():
    global svm_model, cnn_model

    print("Loading SVM Model..")
    svm_model = SVMModel()
    X, y = svm_model.load_data(DATASET_PATH)
    svm_model.train(X, y)
    print("SVM Model loaded.")

    print("Loading CNN Model..")
    cnn_model = CNNModel()
    if cnn_model.load_model():
        labels = sorted([str(i) for i in range(10)] + [chr(i) for i in range(ord('A'), ord('Z')+1)])
        cnn_model.label_map = {label: i for i, label in enumerate(labels)}
        cnn_model.reverse_map = {i: label for label, i in cnn_model.label_map.items()}
        print(f"Label mappings restored: {len(labels)} classes")
    else:
        print("Training CNN Model...")
        X, y, num_classes = cnn_model.load_data(DATASET_PATH)
        cnn_model.train(X, y, epochs=15)
        print("CNN Model trained.")

@app.post('/predict')
async def predict_plate(file: UploadFile = File(...)):
    temp_path = None
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(contents)
            temp_path = tmp.name
        
        plate_img = cv2.imread(temp_path)
        if plate_img is None:
            return {"error": "Failed to read image."}
        
        characters = split_characters(plate_img)

        if not characters:
            return {"error": "Tidak ada karakter terdeteksi"}
        
        svm_result = ""
        svm_confidence = []
        for char_img, aspect_ratio in characters:
            pred, top_preds = svm_model.predict(char_img)
            if pred == '0' and aspect_ratio < 0.35:
                pred = '1'
            elif pred == 'U' and aspect_ratio < 0.35:
                pred = '1'
            svm_result += pred
            svm_confidence.append(1.0)
        
        cnn_result = ""
        cnn_confidence = []
        for char_img, aspect_ratio in characters:
            pred, top_preds, conf = cnn_model.predict(char_img)
            if pred == '0' and aspect_ratio < 0.35:
                pred = '1'
            elif pred == 'U' and aspect_ratio < 0.35:
                pred = '1'
            cnn_result += pred
            cnn_confidence.append(conf / 100.0)  # Normalize to 0-1
        svm_corrected = correct_plate_context(svm_result)
        cnn_corrected = correct_plate_context(cnn_result)
        
        svm_accuracy = sum(svm_confidence) / len(svm_confidence) * 100 if svm_confidence else 0
        cnn_accuracy = sum(cnn_confidence) / len(cnn_confidence) * 100 if cnn_confidence else 0
        overall_accuracy = (svm_accuracy + cnn_accuracy) / 2
        
        best_result = cnn_corrected if cnn_accuracy > svm_accuracy else svm_corrected
        
        return {
            "success": True,
            "hasil": best_result,
            "svm": {
                "result": svm_corrected,
                "accuracy": f"{svm_accuracy:.1f}%"
            },
            "cnn": {
                "result": cnn_corrected,
                "accuracy": f"{cnn_accuracy:.1f}%"
            },
            "overall": f"{overall_accuracy:.1f}%",
            "char_count": len(characters)
        }
        
    except Exception as e:
        return {"error": str(e)}
    
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "svm_loaded": svm_model is not None and svm_model.trained,
        "cnn_loaded": cnn_model is not None and cnn_model.trained
    }
if __name__ == "__main__":
    print("=" * 50)
    print("Plate Recognition API Server")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)

        