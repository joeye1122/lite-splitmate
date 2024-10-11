import os
import json
import base64
import portalocker

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define folder paths for caching
CACHE_FOLDER = os.path.join(ROOT_DIR, "cache", "tenant_photos")
CACHE_FILE = os.path.join(ROOT_DIR, "cache", "cached_data.json")

# Ensure cache folder exists
if not os.path.exists(CACHE_FOLDER):
    os.makedirs(CACHE_FOLDER)

# Ensure cache file exists and has the correct structure
if not os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "w") as f:
        json.dump({"users": [], "utilities": []}, f)


# Load cached users and utilities from the cache file
def load_cached_data():
    with open(CACHE_FILE, "r") as f:
        portalocker.lock(f, portalocker.LOCK_SH)  # Shared lock for reading
        data = json.load(f)
        portalocker.unlock(f)
    return data


# Save tenant photo to the local folder
def save_tenant_photo(tenant_id, base64_photo):
    # Decode the base64 string to binary data
    photo_data = base64.b64decode(base64_photo)
    # Save the photo with the tenant's _id as the filename
    photo_path = os.path.join(CACHE_FOLDER, f"{tenant_id}.jpg")
    with open(photo_path, "wb") as f:
        f.write(photo_data)


# Update the cache file with newly cached tenants and utilities
def update_cache(new_users, new_utilities):
    with open(CACHE_FILE, "r+") as f:
        portalocker.lock(f, portalocker.LOCK_EX)  # Exclusive lock for writing
        cache_data = json.load(f)

        # Update the users and utilities lists in the cache
        cache_data["users"].extend(new_users)
        cache_data["utilities"].extend(new_utilities)

        # Remove duplicates by tenant/utility ID
        cache_data["users"] = remove_duplicates_by_id(cache_data["users"])
        cache_data["utilities"] = remove_duplicates_by_id(cache_data["utilities"])

        f.seek(0)
        json.dump(cache_data, f)
        portalocker.unlock(f)


# Remove stale users from the cache
def remove_stale_users_from_cache(stale_user_ids):
    with open(CACHE_FILE, "r+") as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        cache_data = json.load(f)

        # Filter out stale users
        cache_data["users"] = [
            user for user in cache_data["users"] if user["_id"] not in stale_user_ids
        ]

        # Remove user photos
        for user_id in stale_user_ids:
            photo_path = os.path.join(CACHE_FOLDER, f"{user_id}.jpg")
            if os.path.exists(photo_path):
                os.remove(photo_path)

        # Save updated cache
        f.seek(0)
        json.dump(cache_data, f)
        f.truncate()
        portalocker.unlock(f)


# Remove stale utilities from the cache
def remove_stale_utilities_from_cache(stale_utility_ids):
    with open(CACHE_FILE, "r+") as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        cache_data = json.load(f)

        # Filter out stale utilities
        cache_data["utilities"] = [
            utility
            for utility in cache_data["utilities"]
            if utility["_id"] not in stale_utility_ids
        ]

        # Save updated cache
        f.seek(0)
        json.dump(cache_data, f)
        f.truncate()
        portalocker.unlock(f)


# Helper function to remove duplicates based on "_id"
def remove_duplicates_by_id(data_list):
    unique_data = {
        item["_id"]: item for item in data_list
    }  # Dictionary to remove duplicates
    return list(unique_data.values())
