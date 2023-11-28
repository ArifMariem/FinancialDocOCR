from pymongo import MongoClient
from fastapi import APIRouter
import json
from fastapi import APIRouter, Request 
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi import Depends
from .security import   get_authenticated_user 
from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from starlette.responses import JSONResponse
from bson.binary import Binary
from datetime import datetime
from bson import ObjectId


router = APIRouter()

templates = Jinja2Templates(directory="templates")

def get_database() -> AsyncIOMotorDatabase:
    MONGODB_URL = "mongodb://localhost:27017/"  
    client = AsyncIOMotorClient(MONGODB_URL)
    database = client["Comptes"]  
    return database , client

import os

def get_files_from_media_directory():
    media_dir = "media"
    files = []

    if os.path.exists(media_dir) and os.path.isdir(media_dir):
        files = [os.path.join(media_dir, f) for f in os.listdir(media_dir) if os.path.isfile(os.path.join(media_dir, f))]

    return files


def convert_numbers_to_int(data):
    converted_data = []
    for page in data:
        converted_page = {
            "page_number": page["page_number"],
            "data": [],
            "column_name" : page["column_name"]
        }
        for row in page["data"]:
            converted_row = {}
            for col_name, value in zip(page["column_name"], row):
                if isinstance(value, str):
                    # Remove thousands separator and convert to int if possible
                    value_without_commas = value.replace(',', '')
                    if value_without_commas.isdigit():
                        converted_row[col_name] = int(value_without_commas)
                    else:
                        converted_row[col_name] = value
                else:
                    # Non-string values are kept as is
                    converted_row[col_name] = value
            converted_page["data"].append(converted_row)
        converted_data.append(converted_page)
    return converted_data


    


@router.post("/insert_doc")
async def insert_document(request: Request, doc_data: dict, authenticated_email: str = Depends(get_authenticated_user)):
    data = doc_data['data']
    converted_data = convert_numbers_to_int(data)
    print("converted data is ", converted_data)
    municipalite = doc_data['municipalite']
    date = doc_data['date']

    database, client = get_database()
    files = get_files_from_media_directory()

    collection = database["Document"]
    document = {
        "pages": [],
        "municipality": municipalite,
        "date": date,
        "utilisateur": authenticated_email,
        "pdf_file": []
    }

    for page_data in converted_data:
        page_number = page_data['page_number']
        page_data_list = page_data['data']
        page = {
            "pageNumber": str(page_number),
            "columns": page_data["column_name"] ,
            "rows": page_data_list
            
        }
        document["pages"].append(page)

    for file_path in files:
        with open(file_path, "rb") as file:
            file_content = file.read()

        file_document = {
            "filename": os.path.basename(file_path),
            "content": Binary(file_content)
        }

        document["pdf_file"].append(file_document)

    result = await collection.insert_one(document)

    if result.inserted_id:
        print("Document inserted successfully.")
        for file_path in files:
            os.remove(file_path)
    else:
        print("Failed to insert the document.")

    response_data = {
        "redirect_url": "/get_all_documents",
        "message": "Document inserted successfully"
    }

    return JSONResponse(content=response_data, status_code=200)


async def get_all_documents():
    database, client = get_database()
    collection = database["Document"]
    print("database " , database)

    documents = []
    async for document in collection.find({}):
        documents.append(document)

    return documents


@router.post("/get_all_documents" ,  response_class=HTMLResponse)
async def retrieve_documents(request : Request , authenticated_email: str = Depends(get_authenticated_user)):
    documents =await get_all_documents()
    return templates.TemplateResponse("database.html", {"request": request, "email": authenticated_email , "documents" : documents})


@router.get("/get_all_documents" ,  response_class=HTMLResponse)
async def retrieve_doc(request : Request , authenticated_email: str = Depends(get_authenticated_user)):
    documents = await get_all_documents()
    return templates.TemplateResponse("database.html", {"request": request, "email": authenticated_email , "documents" : documents})

async def get_document_details(document_id: str):
    database, client = get_database()
    collection = database["Document"]
    document_id = ObjectId(document_id)

    document = await collection.find_one({"_id": document_id})

    return document

    


@router.get("/document_details/{document_id}" , response_class=HTMLResponse)
async def document_details(request: Request, document_id: str, authenticated_email: str = Depends(get_authenticated_user)):
    document = await get_document_details(document_id)
    return templates.TemplateResponse("document_details.html", {"request": request, "email": authenticated_email, "document": document})







async def get_all_Users():
    database, client = get_database()
    collection = database["User"]
    print("database " , database)

    users = []
    async for user in collection.find({}):
        users.append(user)

    return users


@router.post("/all_user" ,  response_class=HTMLResponse)
async def retrieve_users(request : Request , authenticated_email: str = Depends(get_authenticated_user)):
    users =await get_all_Users()
    return templates.TemplateResponse("users.html", {"request": request, "email": authenticated_email , "users" : users})


@router.get("/all_user" ,  response_class=HTMLResponse)
async def retrieve_user(request : Request , authenticated_email: str = Depends(get_authenticated_user)):
    users = await get_all_Users()
    return templates.TemplateResponse("users.html", {"request": request, "email": authenticated_email , "users" : users})








