# app/routes.py
from fastapi import APIRouter, File, UploadFile, HTTPException
from PIL import Image
import io
from .machineLearningModel import predict_image
from .scraper import get_upcoming_events

router = APIRouter()


@router.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))

        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')

        predicted_class, predicted_label = predict_image(image)
        return {"predicted_class": predicted_class, "predicted_label": predicted_label}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/test/")
async def test_default_image():
    try:
        default_image_path = 'bandstand_test.jpg'
        image = Image.open(default_image_path)

        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Call the predict_image function
        predicted_class, predicted_label = predict_image(image)

        return {
            "predicted_class": predicted_class,
            "predicted_label": predicted_label
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred during prediction: {str(e)}"
        )


@router.get("/events/")
async def events():
    return await get_upcoming_events()
