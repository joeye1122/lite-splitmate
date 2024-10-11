from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()


# MongoDB connection
def get_db():
    mongoUrl = os.getenv("MONGO_URI")
    if not mongoUrl:
        raise ValueError("MONGO_URI environment variable is not set or empty")
    client = MongoClient(mongoUrl)
    db = client["test"]
    return db


# HOUSE ID
def get_house_id():
    house_id = os.getenv("HOUSE_ID")
    if not house_id:
        raise ValueError("HOUSE_ID environment variable is not set or empty")
    return house_id
