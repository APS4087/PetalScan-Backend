# app/routes.py
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from pydantic import BaseModel
from PIL import Image
import io
from .machineLearningModel import predict_image
from .scraper import get_upcoming_events
from .payment import create_payment_intent, create_subscription

router = APIRouter()


class PaymentRequest(BaseModel):
    amount: int  # Amount in cents
    plan: str  # Plan type


@router.post("/payment-intent/")
async def payment_intent(request: PaymentRequest):
    try:
        if request.plan == 'Monthly':
            intent = await create_subscription(request.amount)
        else:
            intent = await create_payment_intent(request.amount)
        return intent
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
