from config import get_db
from bson import ObjectId
from pymongo.errors import PyMongoError

db = get_db()


# Find utilities by house ID
def find_all_utilities_by_house(house_id):
    try:
        house_id = ObjectId(house_id)
        return db.utilities.find(
            {"houseId": house_id}, {"name": 1, "sensor": 1, "_id": 1}
        )
    except PyMongoError as e:
        raise PyMongoError(
            f"Error fetching utilities for house_id {house_id}: {str(e)}"
        )


# Find tenants by house ID
def find_all_tenants_by_house(house_id):
    try:
        house_id = ObjectId(house_id)
        return db.users.find(
            {
                "$or": [
                    {"houseId": house_id},
                    {"houseId": {"$elemMatch": {"$eq": house_id}}},
                ]
            },
            {"photo": 1, "_id": 1, "name": 1},
        )
    except PyMongoError as e:
        raise PyMongoError(f"Error fetching tenants for house_id {house_id}: {str(e)}")
