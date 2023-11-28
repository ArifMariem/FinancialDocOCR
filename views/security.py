from fastapi import  Depends, Request, Response
from fastapi.templating import Jinja2Templates
from fastapi import Depends, Form, Request,APIRouter
from fastapi.security import HTTPBasic
import secrets
from starlette.responses import RedirectResponse
import hashlib
import os
router = APIRouter()
templates = Jinja2Templates(directory="templates")
security = HTTPBasic()
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

# def authenticate_user(email, password):
#     if email in user_db and user_db[email] == password:
#         return email
#     return None

def get_database() -> AsyncIOMotorDatabase:
    MONGODB_URL = "mongodb://localhost:27017/"  
    client = AsyncIOMotorClient(MONGODB_URL)
    database = client["Comptes"]  
    return database , client

def get_authenticated_user(request: Request):
    session_token = request.cookies.get("session_token")
    email= request.session.get("email")
    return email


def get_authenticated_template(request, template_name, authenticated_email):
    if authenticated_email:
        return templates.TemplateResponse(template_name, {"request": request, "email": authenticated_email})
    else:
        return templates.TemplateResponse("sign-in.html", {"request": request, "error_message": ""})

sessions = {}


# Template directory
templates = Jinja2Templates(directory="templates")

@router.post("/login")
async def login(request: Request, email: str = Form(...), password: str = Form(...)):
    user = await authenticate_user(email, password)
    print("user is ", user)
    emaill = get_authenticated_user(request)
    if user :
        if user["email"]==emaill:
            return templates.TemplateResponse("sign-in.html", {"request": request, "error_message": "User is already connected in another session."})

        session_token = secrets.token_hex(16)
        request.session["session_token"] =session_token
        request.session["email"] = email
        
        response = RedirectResponse(url="/overview")
        response.set_cookie(key="session_token", value=session_token)

       
        return response
    else:
        return templates.TemplateResponse("sign-in.html", {"request": request, "error_message": "Identification invalide"})


@router.get("/login")
async def login(request: Request):
        return templates.TemplateResponse("sign-in.html", {"request": request})


@router.get("/logout")
async def logout(response: Response , request: Request, authenticated_email: str = Depends(get_authenticated_user)):
    authenticated = authenticated_email 
    if "session_token" in request.session:
        del request.session["session_token"]
    if "email" in request.session:
        del request.session["email"]
    response.delete_cookie("session_token")  
    return templates.TemplateResponse("sign-in.html", {"request": request, "error_message": ""})





database , client = get_database()
users_collection = database.get_collection("User")


async def authenticate_user(email, password):
    user = await users_collection.find_one({"email": email})
    if user and verify_password(password, user["password"], user["salt"]):
        return user
    else :
        return None

async def get_user_by_email(email):
    return await users_collection.find_one({"email": email})


async def create_user(username,email, password , salt):
    user = {
        "username" : username ,
        "email": email,
        "password": password, 
        "salt" : salt
    }
    result = await users_collection.insert_one(user)
    return result


@router.post("/signupp")
async def signup(request: Request, email: str = Form(...), password: str = Form(...) ,  username: str = Form(...)):
    existing_user = await get_user_by_email(email)
    
    if existing_user:
        return templates.TemplateResponse("sign-up.html", {"request": request, "error_message": "Utilisateur existe déja."})

   
    hashed_password , salt= hash_password(password)

    await create_user(username,email, hashed_password , salt)

    session_token = secrets.token_hex(16)
    request.session["session_token"] = session_token
    request.session["email"] = email

    response = RedirectResponse(url="/overview")
    response.set_cookie(key="session_token", value=session_token)

    return response






def hash_password(password, salt=None):
    if salt is None:
        salt = os.urandom(16)  

    password_salt = password.encode('utf-8') + salt
    hashed_password = hashlib.sha256(password_salt).hexdigest()

    return hashed_password, salt

def verify_password(provided_password, stored_hashed_password, salt):
    password_salt = provided_password.encode('utf-8') + salt
    hashed_password = hashlib.sha256(password_salt).hexdigest()
    return hashed_password == stored_hashed_password




@router.get("/signup")
async def inscri(request: Request):
        return templates.TemplateResponse("sign-up.html", {"request": request})