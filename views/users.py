from fastapi import APIRouter

router = APIRouter()

@router.get("/users")
async def read_users():
    # Your view function logic for users
    return {"message": "This is the users page"}
