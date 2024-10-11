from flask import Flask, Response, jsonify
from services import find_all_tenants_by_house, find_all_utilities_by_house
from config import get_house_id
import json
from bson import ObjectId
from bson.errors import InvalidId
from pymongo.errors import PyMongoError
from werkzeug.exceptions import HTTPException
from cache import (
    load_cached_data,
    save_tenant_photo,
    update_cache,
    remove_stale_users_from_cache,
    remove_stale_utilities_from_cache,
    CACHE_FILE,
    CACHE_FOLDER,
)
import logging


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return super(JSONEncoder, self).default(obj)


app = Flask(__name__)
app.logger.setLevel(logging.INFO)


# Error handler for general exceptions
@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, HTTPException):
        return jsonify({"error": str(e)}), e.code
    app.logger.error(f"Unhandled exception: {str(e)}")
    return jsonify({"error": "An unexpected error occurred"}), 500


# Error handler for MongoDB-specific errors
@app.errorhandler(PyMongoError)
def handle_mongo_exception(e):
    app.logger.error(f"MongoDB error: {str(e)}")
    return jsonify({"error": "Database error occurred", "details": str(e)}), 500


def validate_object_id(id_str):
    try:
        return ObjectId(id_str)
    except (InvalidId, TypeError):
        raise ValueError("Invalid ObjectId format")


@app.route("/read", methods=["GET"])
def get_users_utilities():
    try:
        # Get house_id from the config
        house_id = validate_object_id(get_house_id())

        # Fetch users and utilities from the database
        try:
            users = list(find_all_tenants_by_house(house_id=house_id))
            utilities = list(find_all_utilities_by_house(house_id=house_id))
        except PyMongoError as e:
            return handle_mongo_exception(e)

        cached_data = load_cached_data()
        cached_users = cached_data["users"]
        cached_utilities = cached_data["utilities"]

        # Convert cached data to dictionaries for faster lookup
        cached_users_dict = {user["_id"]: user for user in cached_users}
        cached_utilities_dict = {
            utility["_id"]: utility for utility in cached_utilities
        }

        # Find new users not in the cache
        new_users = [
            (
                {"_id": str(tenant["_id"]), "name": tenant["name"]}
                if str(tenant["_id"]) not in cached_users_dict
                else None
            )
            for tenant in users
        ]
        new_users = [user for user in new_users if user]  # Filter out None

        # Cache the photos of new users
        for tenant in new_users:
            save_tenant_photo(
                tenant["_id"],
                next(u["photo"] for u in users if str(u["_id"]) == tenant["_id"]),
            )

        # Find new utilities not in the cache
        new_utilities = [
            (
                {
                    "_id": str(utility["_id"]),
                    "name": utility["name"],
                    "mac_address": utility["sensor"],
                }
                if str(utility["_id"]) not in cached_utilities_dict
                else None
            )
            for utility in utilities
        ]
        new_utilities = [
            utility for utility in new_utilities if utility
        ]  # Filter out None

        # Identify stale users and utilities (in cache but not in the database)
        stale_user_ids = set(cached_users_dict.keys()) - {
            str(user["_id"]) for user in users
        }
        stale_utility_ids = set(cached_utilities_dict.keys()) - {
            str(utility["_id"]) for utility in utilities
        }

        # Remove stale users and utilities from cache and delete their photos
        remove_stale_users_from_cache(stale_user_ids)
        remove_stale_utilities_from_cache(stale_utility_ids)

        update_cache(new_users=new_users, new_utilities=new_utilities)

        # Log cache updates
        app.logger.info(
            f"Updated cache with {len(new_users)} new users and {len(new_utilities)} new utilities"
        )

        # Prepare the response data
        response_data = {"cache_json": CACHE_FILE, "image_folder": CACHE_FOLDER}

        # Return a JSON response using the custom encoder
        return Response(
            json.dumps(response_data, cls=JSONEncoder), mimetype="application/json"
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return handle_exception(e)


if __name__ == "__main__":
    app.run(debug=True)
