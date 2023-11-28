import cv2
import numpy as np
from .table_reconstruction import *
from .table_detection import *
from pdf2image import convert_from_path
import asyncio
from fastapi import WebSocket
from typing import Callable
from views.df_check import *
import torch



###Convert pdf to images
def convertpdf2imges(file) : 
    images = convert_from_path(file)
    return images
###convert the layout of a PDF from portrait to landscape
def check_size(images) :
    img=[]
    for i in images :
        if i.size[0]<i.size[1] : 
            img.append(i.rotate(270 , expand=True))
        else:
            img.append(i)
    return img
def compress_i(images):
    img=[]
    for i in images :
        image_array = np.array(i)

    # Convert the image to JPEG format with specific compression quality
        _, compressed_image = cv2.imencode(".jpg", image_array, [cv2.IMWRITE_JPEG_QUALITY, 75])

    # Convert the compressed image back to NumPy array
        decompressed_image = cv2.imdecode(compressed_image, cv2.IMREAD_UNCHANGED)
        img.append(decompressed_image)
    return img



def trait_doc(file) : 
    device = torch.device("cpu")

       
       
    images = convertpdf2imges(file)
    imp= images[1:-2]
    print("doc size is ", len(imp))
    imges = check_size(imp)
    print("check size ", len(imges))
    imags=compress_i(imges)
    print("compress ", len(imags))

    return imags


async def ocr_process(websocket: WebSocket, imags: list, send_progress: Callable, send_dataframe: Callable , model , processor):
    print("len imags is ", len(imags))
    doc = {}
    

    async def update_progress(k: int, total_pages: int):
        progress = int((k) / total_pages * 100)
        await send_progress(progress)
    async def update_dataframe(page_number: int, dataframe: pd.DataFrame):
        await send_dataframe(page_number, dataframe)
    for k in range(len(imags)) :
        
            print("page num " , k) 
            




            if k ==1 :
            
                tab = imags[k].copy()
                img_bin , img, kernel = preprocess_table(tab)
                
                images , box  , result =table_detection(img_bin , img, kernel)
            
                closing , gray = preprocess(result)
                img2 , postions = line_detection_1(closing , result )
                imgi ,postion_img , numo , postion_num = extract_writing (img2 , postions) 
                one_third = len(imgi) // 3

                # Use slicing to get one-third of the list
                one_third_images = imgi[:one_third]
                                
                arabic_img = arabic_cells(one_third_images)
                ind = []
                for j in arabic_img :
                    ind.append(j[1])
                merged_res = imgi+ numo
                    # Convert the image to BGR format
                res=[]
                for imi in numo :
                    bgr_image = Image.fromarray(imi)
                    res.append(bgr_image)
                
                merged_pos = postion_img +postion_num
            
                cell_data =text_placement(postion_img , res , [] , arabic_img , ind , model , processor )
                columns , rows = column_row(postion_img , 35 , 35) 
                
                
            
                
            else :
                
                
                tab = imags[k].copy()
            
                img_bin , img, kernel = preprocess_table(tab)
                boxes , res , index_empty , new_new_box , arabic_img , indi =cell_detection(img_bin , img, kernel , k )
                cell_data =text_placement(boxes , res , index_empty ,[] , indi , model , processor )
                ###threshold for page 3 and more is 30 
                columns , rows  = column_row(boxes , 25 , 30)
                print("rows are ", rows)
                
                if k == 0 :
                    news = new_rows(rows)
                    new_rowss = flatten_rows(news) 
                    rows = merge_consecutive_rows(new_rowss , 25)
            df = table_rec(columns, rows , cell_data)
            await update_progress(k, len(imags))
            print("df is " , df)
            
            
            
            await update_dataframe(k , df)

            
            
            doc[k] = df 

    await update_progress(len(imags), len(imags))


    

    return doc
        
            