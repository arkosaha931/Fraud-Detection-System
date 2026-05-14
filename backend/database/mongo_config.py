from pymongo import MongoClient

MONGO_URI = "mongodb+srv://bersalonalm10_db_user:arko%40007@cluster0.pkp9zaz.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

client = MongoClient(MONGO_URI)

db = client["fraudlens"]

accounts_collection = db["accounts"]

transactions_collection = db["transactions"]