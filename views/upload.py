from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from fastapi import Depends
from starlette.status import HTTP_302_FOUND
from fastapi.responses import FileResponse
from urllib.parse import unquote
from .security import get_authenticated_template , get_authenticated_user 
from datetime import date

from fastapi import Form, File, UploadFile, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
import os
from pathlib import Path


router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/upload_pdf/", response_class=HTMLResponse)
async def show_upload_pdf_form(request: Request,authenticated_email: str = Depends(get_authenticated_user)):
    authenticated = authenticated_email is not None
    if authenticated_email is not None :
        return get_authenticated_template(request, "upload.html", authenticated_email)
    else :
        return templates.TemplateResponse("sign-in.html", {"request": request, "error_message": "Identification invalide"})


@router.get("/media/{file_name}", name="serve_pdf")
async def serve_pdf(file_name: str):
    pdf_path = Path("media") / file_name
    return FileResponse(pdf_path, media_type="application/pdf")

@router.post("/upload_pdf/", response_class=HTMLResponse)
async def upload_pdf_view(request: Request, pdf_file: UploadFile = File(...) ,  municipalite: str = Form(...), date: date = Form(...)):
    file_url = None
    if pdf_file:
        # Process the uploaded file
        fs = Path("media")
        filename = fs / pdf_file.filename
        try:
            with filename.open("wb") as f:
                f.write(pdf_file.file.read())

            # Generate the absolute URL for the PDF file using request.url_for
            file_url = request.url_for("serve_pdf", file_name=filename.name)
        except Exception as e:
            return templates.TemplateResponse("upload.html", {"request": request, "file_url": None, "error": str(e)})
    return templates.TemplateResponse("upload.html", {"request": request, "file_url": file_url , "municipalite" : municipalite , "date" : date})

@router.post("/validate_pdf/")
async def validate_pdf(request: Request, filename: str = Form(...), municipalite: str = Form(...), date: date = Form(...)):
    filename = unquote(filename)
    if filename.startswith("http://127.0.0.1:8000/"):
        filename = filename[len("http://127.0.0.1:8000/"):]
    base_filename = os.path.basename(filename)
    temp_filepath = filename
    new_filename = 'validated_' + base_filename 
    new_filepath = "media" +"/"+ new_filename
    
    os.rename(temp_filepath, new_filepath)
    cleanup_files()
    return RedirectResponse(url=f"/extraction?filename={new_filename}&municipalite={municipalite}&date={date}", status_code=HTTP_302_FOUND)




def cleanup_files():
    media_directory = Path("media")  # Replace with the actual path to your media directory
    validated_files = []
    for file_path in media_directory.iterdir():
        if file_path.is_file() and not file_path.name.startswith('validated_'):
            validated_files.append(file_path)
        else:
            # Handle any additional filtering logic if needed
            pass

    # Perform cleanup for files that don't have 'validated_' prefix
    for file_path in validated_files:
        try:
            file_path.unlink()  # Remove the file
        except OSError as e:
            # Handle any exceptions if required
            raise HTTPException(status_code=500, detail="Error cleaning up files")

    return len(validated_files)  # Return the number of files cleaned up
