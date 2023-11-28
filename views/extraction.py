from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
import websockets
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
import json
import queue
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.status import HTTP_302_FOUND

from datetime import date
from OCR_PROCESS.full_script import *
import ast
from fastapi import Depends

from fastapi.responses import HTMLResponse
from fastapi import WebSocket

import os
import asyncio
from fastapi import BackgroundTasks
from .security import get_authenticated_template , get_authenticated_user 

import threading
router = APIRouter()
templates = Jinja2Templates(directory="templates")




from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import TrOCRProcessor

model = VisionEncoderDecoderModel.from_pretrained("./OCR_PROCESS/Model/saved_final")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")


@router.get("/extraction", response_class=HTMLResponse)
async def extraction(request: Request, filename: str, municipalite: str, date: date, authenticated_email: str = Depends(get_authenticated_user)):
    if authenticated_email is not None : 
            
        websocket_url = f"ws://127.0.0.1:8000/ws?filename={filename}&municipalite={municipalite}&date={date}"
        
        async with websockets.connect(websocket_url) as websocket:
            await websocket.send("Start OCR")

        return templates.TemplateResponse("extraction.html", {"request": request, "filename": filename, "email": authenticated_email, "municipalite": municipalite, "date": date})
    else : 
        return templates.TemplateResponse("sign-in.html", {"request": request, "error_message": "Identification invalide"})



@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket  , background_tasks: BackgroundTasks):
    await websocket.accept()
    async def run_ocr_process(websocket, imags):
        global doc

        doc = await ocr_process(websocket, imags, send_progress, send_dataframe, model, processor)
        doc_ready.set()

    doc_ready = asyncio.Event()
    async def send_progress(progress: int):
                    data1 = {"type": "progress", "progress": progress}
                    await websocket.send_text(json.dumps(data1))

    async def send_dataframe(page_number: int, dataframe: pd.DataFrame):
                    threshold = 0.3* len(dataframe.columns)

                    # Drop rows with more than the threshold number of null values
                    cleaned_dataframe = dataframe.dropna(thresh=threshold)
                    df = cleaned_dataframe
                    print("cleaned data ", df)
                    if page_number!=1 :
                        new_df = add_columns(df , page_number)
                    else : 
                        new_df = df
                    new_df.fillna(0, inplace=True)
                    if page_number != 0 :
                        new_df = add_details(new_df , page_number)
                    new_df = remove_columns_if_exist(new_df)
                    if page_number != 0 and  page_number!=1  and page_number !=16  and page_number !=17  and page_number !=34  and page_number !=35:
                        new_df = add_ref(new_df , page_number)

                    print("new_df is " , new_df)
                    data2 = {"type": "dataframe", "page_number": page_number, "dataframe":  new_df.to_json()}
                    await websocket.send_text(json.dumps(data2))


    def background_ocr_process(websocket, imags):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_ocr_process(websocket, imags))
        loop.close()
    doc = None

    try:
        while True:

            data = await websocket.receive_text()
            if data.startswith("filename:"):
                parameters = data.replace("filename:", "").strip().split("&")
                filename = parameters[0].replace("filename=", "")
                print("the received filename ,=", filename)

                full_path = os.path.join("media", filename)
                imags = trait_doc(full_path)

                thread = threading.Thread(target=background_ocr_process, args=(websocket, imags))
                thread.start()   

                
            if data.startswith("corrected_data:"):
                    corrected_data, page_number = data.replace("corrected_data:", "").split(":")

                    data = json.loads(corrected_data)
                   

                   
                    check_results= await check_dataframe(data , int(page_number) )
                    data2 = {"type": "check_results", "result": check_results}
                    await websocket.send_text(json.dumps(data2))
            
            
            
    except WebSocketDisconnect:
        pass



