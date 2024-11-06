from fastapi import Form, FastAPI, HTTPException, Depends
import logging
from .routes import router as api_router
from .scraper import get_upcoming_events  # Import the scraping function
import asyncio
from firebase_admin import auth
from app.firebase_config import *

# Configure logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Include the API routes
app.include_router(api_router)

# Dependency to check if the user is an admin


def verify_admin(admin_email: str):
    user = auth.get_user_by_email(admin_email)
    if not user.custom_claims.get("admin"):
        raise HTTPException(status_code=403, detail="Not authorized")
    return user

# Endpoint to update user email and password


@app.post("/update-user")
async def update_user(
    target_email: str = Form(...,
                             description="The email of the user to be updated"),
    new_email: str = Form(None, description="The new email for the user"),
    new_password: str = Form(
        None, description="The new password for the user"),
    admin_email: str = Form(..., description="The email of the admin user"),
    admin_user: auth.UserRecord = Depends(verify_admin)
):
    logging.info(
        f"Received request to update user: target_email={target_email}, new_email={new_email}, new_password={new_password}, admin_email={admin_email}"
    )

    try:
        user = auth.get_user_by_email(target_email)
        updates = {}
        if new_email:
            updates["email"] = new_email
        if new_password:
            updates["password"] = new_password
        auth.update_user(user.uid, **updates)
        return {"message": f"User {target_email} updated successfully"}
    except auth.AuthError as e:
        logging.error(f"Firebase Auth error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Firebase Auth error: {e}")
    except Exception as e:
        logging.error(f"Error updating user: {e}")
        raise HTTPException(status_code=500, detail="Error updating user")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
