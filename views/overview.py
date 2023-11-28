from fastapi import APIRouter, Request 
from fastapi.templating import Jinja2Templates
from fastapi import Depends, Request
from fastapi.responses import HTMLResponse
from .security import get_authenticated_template , get_authenticated_user 


router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.get("/overview", response_class=HTMLResponse)
async def get_overview(request: Request, authenticated_email: str = Depends(get_authenticated_user)):
    authenticated = authenticated_email 
    return get_authenticated_template(request, "overview.html", authenticated_email)    


@router.post("/overview", response_class=HTMLResponse)
async def post_overview(request: Request):
    authenticated_email = get_authenticated_user(request)
    if authenticated_email is not None :
        return templates.TemplateResponse("overview.html", {"request": request, "email": authenticated_email})
    
    else :
        return templates.TemplateResponse("sign-in.html", {"request": request, "error_message": "Identification invalide"})

