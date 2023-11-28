import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import easyocr
import re
from multiprocessing import Manager
import multiprocessing
from functools import partial
import io



from pdf2image import convert_from_path
import cv2

def compress_img(img):
    # Convert the image to JPEG format with specific compression quality
    _, compressed_image = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])

# Convert the compressed image back to NumPy array
    decompressed_image = cv2.imdecode(compressed_image, cv2.IMREAD_UNCHANGED)
    return decompressed_image

def simulate_save_and_open(image):
    # Convert image to RGB color space

    # Create an in-memory file object
    image_file = io.BytesIO()

    # Save the RGB image to the in-memory file object as JPEG with specific compression quality
    image.save(image_file, format='JPEG', quality=80)

    # Rewind the file object to the beginning
    image_file.seek(0)

    # Read the in-memory file object as a JPEG image using OpenCV
    decompressed_image = cv2.imdecode(np.frombuffer(image_file.getvalue(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    return decompressed_image

def preprocess_table(img) : 
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5),np.uint8)
    thresh,img_bin = cv2.threshold(img,220,255,cv2.THRESH_BINARY)
    img_bin = 255-img_bin
    
    return img_bin , img, kernel

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def table_detection(img_bin , img , kernel) :
    #detecting horizontal and vertical lines in image
    horizontal = np.copy(img_bin)
    vertical = np.copy(img_bin)
     # Specify size on horizontal axis
    cols = horizontal.shape[1]
    horizontal_size = cols // 30
    # structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    # Show extracted horizontal lines
    # Specify size on vertical axis
    rows = vertical.shape[0]
    verticalsize = rows // 30
    #structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)


    img_vh = cv2.bitwise_or(horizontal, vertical)

    lines = cv2.HoughLinesP(img_vh, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # Remove the detected lines from the original image
    result = np.copy(img)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(result, (x1, y1), (x2, y2), (255, 255, 255), 3)  # Draw white lines over the original image to eliminate them
    imge=result.copy()
    contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")
    box = []
    # Get position (x,y), width and height for every contour and show the contour on image
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w<950 and w > 50 and h > 50 and h<2000):
            image = cv2.rectangle(result,(x,y),(x+w,y+h),(255,0,0),-1)
            box.append([x,y,w,h])
    images=[]
    for i in box:
        x=i[0]
        y=i[1]
        w=i[2]
        h=i[3]
        imgg1= imge[ y-2:y+h+11,  x-6:x+w+7] 
        images.append(imgg1)
    return images , box  , result


def contains_arabic_words(word_list):
    arabic_pattern = re.compile(r'[\u0621-\u064A]')
    arabic_word_count = 0
    for word in word_list:
        arabic_character_count = len(re.findall(arabic_pattern, word))
        if arabic_character_count > len(word) / 2: #define threshold for words in arabic, if more then half is considerate arabic then all word is.
            arabic_word_count += 1
    return arabic_word_count
def process_ocr(args ):
    image_path, i = args
    arabic_img=[]
    arabic_text=[]
    reader = easyocr.Reader(['ar'] , gpu = False)
    result = reader.readtext(image_path) #apply easyocr on images
    for detection in result:
        if detection[2]>= 0.5: #confidance score for arabic words.
            arabic_text.append(detection[1])
    arabic_count = contains_arabic_words(arabic_text)
            
    if arabic_count >=1:
        arabic_img.append([image_path, i , arabic_text])
        i += 1


    return arabic_img
    

def arabic_cells(images):

    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=6) #define needed cores of cpu for process

    partial_process_ocr = partial(process_ocr)
    inputs = [(image, i) for i, image in enumerate(images)]
    results = pool.map(partial_process_ocr, inputs)

    pool.close()
    pool.join()

    arabic_img = [result for sublist in results for result in sublist]

    return arabic_img
def sort_regions_by_x(regions):
    # Define a key function that extracts the x-coordinate of each region
    def get_x_coord(region):
        x, y, w, h = cv2.boundingRect(region)
        return x

    # Sort the regions by their x-coordinate using the key function
    sorted_regions = sorted(regions, key=get_x_coord)

    return sorted_regions
def preprocess(image):
    thresh,gray = cv2.threshold(image,200,255,cv2.THRESH_BINARY)
    kernel = np.ones((3,4),np.uint8)
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=1)

    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closing , gray
def filter_regions(sorted_regions) :
      filtered_regions=[]
      all_h=[]
      all_y=[]
      all_x=[]
      if len(sorted_regions) != 0 : 
            
            for region in sorted_regions:
                  # Extract bounding box coordinates
                  x, y, w, h = cv2.boundingRect(region)
                  all_y.append(y)
                  all_h.append(y+h)
                  all_x.append(x)
            avg_h=sum(all_h)/len(sorted_regions)
            avg_y=sum(all_y)/len(sorted_regions)
            yy=avg_y
            for region in sorted_regions:
                  # Extract bounding box coordinates
                  x, y, w, h = cv2.boundingRect(region)
                  if abs(y-yy)<=avg_h/2 :
                        filtered_regions.append(region)

      return filtered_regions , all_h , all_y
def regions_detection(closing , img  , gray) :
    img2=None
    imgi=[]
    positions=[]
    list_empty=[]

    mser = cv2.MSER_create(min_area=50
                       ,max_area=600)
    # Detect MSER regions in the image
    regions, _ = mser.detectRegions(closing)
    sorted_regions=sort_regions_by_x(regions)
    filtered , all_h , all_y = filter_regions(sorted_regions)
    if len(filtered) != 0 : 
        digits=[]
    
        for region in filtered:
                # Extract bounding box coordinates
            x, y, w, h = cv2.boundingRect(region)
            digits.append([x,y,w,h])
        y =[subarray[1] for subarray in digits]
        numbers=sorted(all_y)
        heights =sorted(all_h)
        results = {}
        results1={}
            # initialize the key and values for the first result
        key = str(numbers[0])
        key1 = str(heights[0])
        value = [numbers[0]]
        value1 = [heights[0]]

        results[key] = value
        results1[key1] = value1
            # iterate through the list of numbers and compare adjacent numbers
        for i in range(len(numbers)-1):
            diff = abs(numbers[i+1] - numbers[i])
            if diff > 25:
                    # create a new key and add the current number to its value list
                    key = str(numbers[i+1])
                    key1= str(heights[i+1])
                    value = [numbers[i+1]]
                    value1 = [heights[i+1]]
                    results[key] = value
                    results1[key1] = value1
            else:
                    # add the current number to the previous key's value list
                    value.append(numbers[i+1])
                    results[key] = value


                    value1.append(heights[i+1])
                    results1[key1] = value1


                    
        res_val=list(results.values())
        res_val1=list(results1.values())

        for i in range(len(res_val)):

            y1 = res_val[i][0]
            y2 = res_val[i][len(res_val[i]) -1]
            if y2 - y1 <5:
                    continue

            else:
                    y2=res_val1[i][len(res_val1[i]) -1]
                    x1 = closing.shape[1]
                    if y1<8 :
                        y1=8
                    x = 0
                    y = y1 - 8
                    w = x1 - x - 1
                    h = y2 - y1 + 12
                    positions.append([x,y ,w,h])
                    imgip=np.array(img)
                    imgi.append([imgip[y1-8:y2 + 12, 0:x1 - 1]])
        img2=imgi


    if len(positions)!=0 :
        remaining_regions = extract_empty_regions(positions , gray)
        for pos in remaining_regions : 
            if pos[3] >30 :
                    imgx = gray[pos[1]: pos[3] +pos[1], pos[0]:pos[2]+pos[0]]
                    
                    list_empty=empty_cells(imgx , pos)

            else:
                continue
    if len(positions)==1 and len(list_empty)==0 : 
        x= 0
        y=0
        w = img.shape[1]
        h = img.shape[0]
        positions = [[x,y,w,h]]
        img2 =[ [img[y:y +h, x:x + w]]]
    if len(positions) == 0 :
        pos =[ 0 , 0 , img.shape[1] , img.shape[0]]
        list_empty=empty_cells(gray , pos)
        img2=[[]]


    if img2==None : 
        img2=[[]]

    
                    
    return [img2 , positions , list_empty ]
def line_detection_1(closing , img ):
    mser = cv2.MSER_create(min_area=100
                       ,max_area=600)
# Detect MSER regions in the image
    regions, _ = mser.detectRegions(closing)
    threshold=50
    #first_y=avg_y
    filtered_regions=[]
    sorted_regions=sort_regions_by_x(regions)
    all_h=[]
    all_x=[]
    all_y=[]
    for region in sorted_regions:
        # Extract bounding box coordinates
        x, y, w, h = cv2.boundingRect(region)
        all_y.append(y)
        all_h.append(y+h)
        all_x.append(x)
    avg_h=sum(all_h)/len(sorted_regions)
    avg_y=sum(all_y)/len(sorted_regions)
    yy=avg_y
    print(yy)
    filtered_regions=[]
    for region in sorted_regions:
        # Extract bounding box coordinates
        x, y, w, h = cv2.boundingRect(region)
        
        if abs(y-yy)<=avg_h/2 :
            filtered_regions.append(region) 
    digits=[]
    for region in filtered_regions:
    # Extract bounding box coordinates
        x, y, w, h = cv2.boundingRect(region)
        digits.append([x,y,w,h])
    y =[subarray[1] for subarray in digits]
    numbers=sorted(all_y)
    heights =sorted(all_h)
    results = {}
    results1={}

    # initialize the key and values for the first result
    key = str(numbers[0])
    key1 = str(heights[0])

    value = [numbers[0]]
    value1 = [heights[0]]

    results[key] = value
    results1[key1] = value1

    # iterate through the list of numbers and compare adjacent numbers
    for i in range(len(numbers)-1):
        diff = abs(numbers[i+1] - numbers[i])
        if diff > 20:
            # create a new key and add the current number to its value list
            key = str(numbers[i+1])
            key1= str(heights[i+1])
            value = [numbers[i+1]]
            value1 = [heights[i+1]]
            results[key] = value
            results1[key1] = value1
        else:
            # add the current number to the previous key's value list
            value.append(numbers[i+1])
            results[key] = value
            
        
        value1.append(heights[i+1])
        results1[key1] = value1
    res_val=list(results.values())
    res_val1=list(results1.values())
    img2=[]
    position=[]
    for i in range(len(res_val)):
        y1 = res_val[i][0]
        y2 = res_val[i][len(res_val[i]) -1]
        if y2 - y1 <4:
            continue

        else:

            y2=res_val1[i][len(res_val1[i]) -1]
            x1 = img.shape[1]
            if y1<8 :
                y1=8
            img2.append(img[y1-8:y2 + 15, 3:x1 - 1])
            x = 0
            y = y1 - 8
            w = x1 - x - 1
            h = y2 - y1 + 11
            position.append([x,y ,w,h])
    return img2 , position

def extract_writing (img2 , position) :
    imgi=[]
    numo=[]
    postion_img=[]
    postion_num=[]
    for i in range(len(img2)) :
        ar = img2[i][0:img2[i].shape[0] ,img2[i].shape[1]//3  : img2[i].shape[1] ]
        num=img2[i][0:img2[i].shape[0] ,  100: img2[i].shape[1]//3 -100]
        imgi.append(ar)
        x,y,w,h = position[i]
        postion_img.append([x+100 , y ,img2[i].shape[1]//3 -100, img2[i].shape[0] ])
        numo.append(num)
        postion_num.append([x+img2[i].shape[1]//3 , y , img2[i].shape[1] , img2[i].shape[0] ])
    return imgi ,postion_img , numo , postion_num


def fill_indi(ind) :
        indi=[]
        for i in range(len(ind) - 1):
            current_value = ind[i]
            next_value = ind[i + 1]
            indi.append(current_value)
            
            if next_value - current_value > 1:
                # Fill the missing values
                for j in range(current_value + 1, next_value):
                    indi.append(j)

        indi.append(ind[-1])  # Append the last value
        return indi

def cell_detection(img_bin , img, kernel , k ) :
     

    cells , box , result = table_detection(img_bin , img , kernel)
    if k ==0 : 

      
                                    
        arabic_img = arabic_cells(cells)
        
    else:
        one_third = len(cells) // 3

                    # Use slicing to get one-third of the list
        one_third_images = cells[:one_third]
                                    
        arabic_img = arabic_cells(one_third_images)
        

    k=0
    ind = []
    for j in arabic_img :
        ind.append(j[1])
    if ind[0] != 0 :
        ind.insert(0,0)
    
    if k == 0 : 
        indi=ind
    else:
        indi=fill_indi(ind)
    imes=[]
    list_ind=[]
    for i in range(len(cells)):
        if i in indi : 
            continue 
        else:
            
            img=cells[i].copy()
           
            
            img= Image.fromarray(img)
            imgx = simulate_save_and_open(img)
           
            closing , gray = preprocess(imgx)
            imes.append(regions_detection(closing ,imgx , gray))
            list_ind.append(i)
            if i == 16 : 
                print("imes is " ,regions_detection(closing ,imgx , gray) )
        
    res = []
    new_new_box=[]
    boxes=[]
    index_empty=[]
    
    s=-1
    ss= indi[-1]+1
   
    i=0
    while (i < len(imes)):
        ii = list_ind[i]
        if len(imes[i][0][0]) > 0 and len( imes[i][0][0][0]) !=0:
            
            original_coords = box[ii]
            cropped_images = imes[i][0]
            boxess=imes[i][1]
            num_crops = len(cropped_images)

            # Calculate the coordinates for each cropped image
            cropped_coords = []
            x1 = original_coords[0]
            y1=i
            for j in range(num_crops):

                [x, y,w,h ]= boxess[j]
                
                new_new_box.append([box[ii][0] + x,box[ii][1]+ y, box[ii][2] , h])

                boxes.append([box[ii][0]+x,box[ii][1]+ y, w, h])
                if num_crops >=1 :

                    img=Image.fromarray(imes[y1][0][j][0])
                    s=s+1

                    res.append(img)
                
            if len(imes[i][2]) !=0:
                for j in range(0,len(imes[i][2])):
                   
                    
                    [x, y,w,h ]= imes[i][2][j]
                    new_new_box.append([box[ii][0] + x,box[ii][1]+ y, box[ii][2] , h])
                    boxes.append([box[ii][0] + x,box[ii][1]+ y,  w , h])

                    img=cells[ii].copy()
                    imgi=Image.fromarray(img[y:h+y , x:w+x])
                    s=s+1
                    index_empty.append(s)


                    res.append(imgi)

        
        if len(imes[i][0][0]) ==0 :
            if len(imes[i][2]) ==0 : 
            
                boxes.append(box[ii])
                new_new_box.append(box[ii])
                img=Image.fromarray(cells[ii])
                s=s+1
                index_empty.append(s)

                res.append(img)
                
            if len(imes[i][2]) !=0:
                for p in range(len(imes[i][2])) :
                    [x, y,w,h ]= imes[i][2][p] 
                    img=cells[ii].copy()
                    h1,w1 = img.shape
     
                    new_new_box.append([box[ii][0] + x,box[ii][1]+ y,w1, h1])
                    boxes.append([box[ii][0] + x,box[ii][1]+ y,  w1 , h1])
                    imgi=Image.fromarray(img)
                    s=s+1
                    index_empty.append(s)
                    res.append(imgi)
            
        i=i+1
    
    
    return boxes , res , index_empty , new_new_box , arabic_img , indi

def check_key_condition(points, image_width):
    sorted_points = sorted(points, key=lambda p: p[0])  # Sort points by x-coordinate

    first_point_x = sorted_points[0][0]
    last_point_x = sorted_points[-1][0]

    difference_x = last_point_x - first_point_x
    half_image_width = image_width / 2

    if difference_x > half_image_width:
        return True
    else:
        return False       
    
def empty_cells(imgx , pos) : 
  empty=[]

  
  kernel = np.ones((3,3),np.uint8)

  thresh,gray = cv2.threshold(imgx,220,255,cv2.THRESH_BINARY)
  result = imgx.copy()
  horizontal = np.copy(gray)
          # Specify size on horizontal axis
  cols = horizontal.shape[1]
  horizontal_size = cols // 30
          # Create structure element for extracting horizontal lines through morphology operations
  horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
          # Apply morphology operations
  horizontal = cv2.erode(horizontal, horizontalStructure)
  horizontal = cv2.dilate(horizontal, horizontalStructure)
              # Show extracted horizontal lines
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
  morph = cv2.morphologyEx(horizontal, cv2.MORPH_CLOSE, kernel)
  y_coordinates = np.where(morph == 0)[0]
  x_coordinates = np.where(morph == 0)[1]

  indices = np.argsort(y_coordinates)
  sorted_ally = y_coordinates[indices]
  sorted_allx = x_coordinates[indices]
  


  results = {}
  if len(sorted_ally) !=0:
        key = str(sorted_ally[0])
        value = [(sorted_allx[0], sorted_ally[0])]
        results[key] = value

    # Iterate through the sorted y-coordinates and their corresponding x-coordinates
        for i in range(len(sorted_ally) - 1):
            diff = abs(sorted_ally[i + 1] - sorted_ally[i])
            if diff > 30:
                # Create a new key and add the current coordinates to its value list
                key = str(sorted_ally[i + 1])
                value = [(sorted_allx[i + 1], sorted_ally[i + 1])]
                results[key] = value
            else:
                # Add the current coordinates to the previous key's value list
                value.append((sorted_allx[i + 1], sorted_ally[i + 1]))
                results[key] = value
            

        x_threshold = cols // 2  # Threshold for maximum x-coordinate
        y_threshold = 80  # Threshold for length of value lists
        filtered_results = {}

        for key, value in results.items():
            if len(value) >= y_threshold and check_key_condition(value, imgx.shape[1]):
                filtered_results[key] = value

        # Get the keys of filtered_results
        filtered_keys = [int(key) for key in filtered_results.keys()]
                
        x,y,w,h =pos
        for i in filtered_keys :
                if i < 40:
                       i=40
                                
                empty1 = result[ i-40 : i+40 , 0:imgx.shape[1]]
                empty.append( [0,y+i-40, empty1.shape[1] , empty1.shape[0] ])

  
  return empty


def extract_empty_regions(positions , gray ):
    sorted_positions = sorted(positions, key=lambda pos: pos[1])

    remaining_spaces = []
    prev_bottom = 0  # Track the bottom y-coordinate of the previous image

    for pos in sorted_positions:
        top = pos[1]  # Top y-coordinate of the current image
        remaining_space = top - prev_bottom

        if remaining_space > 0:
            remaining_spaces.append((prev_bottom, top))
        prev_bottom = pos[1] + pos[3]  # Update the bottom y-coordinate of the previous image

    # Handle the remaining space after the last image
    last_bottom = sorted_positions[-1][1] + sorted_positions[-1][3]
    remaining_space = gray.shape[0] - last_bottom
    if remaining_space > 0:
        remaining_spaces.append((last_bottom, gray.shape[0]))
    remaining_regions_info = []

    for space in remaining_spaces:
        top, bottom = space
        x = 0  # Leftmost x-coordinate
        y = top  # Top y-coordinate
        w = gray.shape[1]  # Width of the image
        h = bottom - top  # Height of the remaining space

        remaining_regions_info.append([x, y, w, h])

    return remaining_regions_info