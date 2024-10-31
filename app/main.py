# app/main.py
from fastapi import FastAPI
from .routes import router as api_router
from apscheduler.schedulers.background import BackgroundScheduler
from .scraper import get_upcoming_events  # Import the scraping function
import asyncio

app = FastAPI()

# Include the API routes
app.include_router(api_router)

# Initialize the scheduler
scheduler = BackgroundScheduler()

# Define the scheduled scraping job


@scheduler.scheduled_job("interval", hours=24)  # Run every 24 hours
def scheduled_scraping_job():
    try:
        # Run the async scraping function in the event loop
        asyncio.run(get_upcoming_events())
    except Exception as e:
        print(f"Error occurred during scheduled scraping: {e}")

# Start the scheduler when the app starts


@app.on_event("startup")
async def startup_event():
    scheduler.start()
    print("Scheduler started!")

# Shutdown scheduler when the app shuts down


@app.on_event("shutdown")
def shutdown_event():
    scheduler.shutdown()
    print("Scheduler stopped!")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
