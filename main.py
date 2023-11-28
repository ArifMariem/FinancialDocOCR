from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from fastapi import FastAPI

from views.overview import router as overview_router
from views.users import router as user_router
from views.upload import router as upload_router
from views.extraction import router as extraction_router
from views.security import router as security_router
from views.mongo import router as mongorouter

from fastapi import FastAPI, Depends
from views.security import get_authenticated_template , get_authenticated_user


app = FastAPI()


from starlette.middleware.sessions import SessionMiddleware

app.add_middleware(SessionMiddleware, secret_key="some-random-string")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
app.mount("/media" , StaticFiles(directory="media"), name="media")
app.include_router(overview_router)
app.include_router(user_router)
app.include_router(upload_router)
app.include_router(extraction_router)
app.include_router(security_router)
app.include_router(mongorouter)




@app.get("/")
async def index(request :Request ,authenticated_email: str = Depends(get_authenticated_user)):
    authenticated = authenticated_email
    
    return get_authenticated_template(request, "overview.html", authenticated)
        
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

