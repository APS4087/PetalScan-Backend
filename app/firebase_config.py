# app/firebase_config.py
import firebase_admin
from firebase_admin import credentials

# Path to your Firebase service account key JSON file
cred = credentials.Certificate(
    "petalscan-60af5-firebase-adminsdk-3cfpb-5a4a68d1a7.json")
firebase_admin.initialize_app(cred)
