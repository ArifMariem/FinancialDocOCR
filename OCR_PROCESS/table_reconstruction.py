from .table_detection import *
from .trocr import *
from multiprocessing import Pool



def text_placement(boxes , res , index_empty , arabic_img , ind , model , processor ) : 
    pooli = multiprocessing.Pool(processes=6)

    text = [None] * len(res)

    inputs=[]
    for i, r in enumerate(res):
            

            if i in index_empty:
                text[i]= '0'
            else:
                img_pr = preprocess_data(r)
                inputs.append(( model, processor ,  img_pr,i))

    with pooli as pool:
            results = pool.map(inference_model, inputs)
    for generated_text ,i in results:
        text[i] = generated_text
        
 
    cell_data = []

    for i  in range(len(boxes)):
        [x, y, w, h] = boxes[i]
        cell_text = text[i] # Set the OCR text for the cell if available
        print("the text isssss" , text[i])
        cell_data.append((x,y,w,h, cell_text))
    return cell_data

# Define the thresholds for column and row grouping
def column_row(new_new_box , x_threshold,y_threshold) : 
    box_coordinates = new_new_box

    columns = []
    rows = []

    # Group cells into columns and rows
    for i, box in enumerate(box_coordinates):
        x, y, w,h = box
        column_found = False
        row_found = False

        # Check if the cell belongs to an existing column based on x-coordinate proximity
        for column in columns:
            col_index, col_box = column[0]
            col_x, _, _, _ = col_box
            if abs(x - col_x) <= x_threshold:
                column.append((i, box))
                column_found = True
                break
            
        # Check if the cell belongs to an existing row based on y-coordinate proximity
        for row in rows:
            row_index, row_box = row[0]
            _, row_y, _, row_h = row_box
            if abs(y - row_y) <= y_threshold :
                    row.append((i, box))
            
                    row_found = True
                    break

        # If the cell does not belong to any existing column, create a new column
        if not column_found:
            columns.append([(i, box)])

        # If the cell does not belong to any existing row, create a new row
        if not row_found:
            rows.append([(i, box)])

    columns.sort(key=lambda col: col[0][1][0])
    rows.sort(key=lambda col: col[0][1][1])

    
    return columns, rows
def truncate_rows(rows, max_columns):
    all_rows_under_max = all(len(row) <= max_columns for row in rows)

    while not all_rows_under_max:
        for i in range(len(rows)):
            if len(rows[i]) > max_columns:
                excess_cells = rows[i][max_columns:]  # Get the excess cells
                rows[i] = rows[i][:max_columns]  # Truncate the row to the maximum number of columns

                # Add a new row with the exceeding cells just after the current row
                rows.insert(i + 1, excess_cells)

        all_rows_under_max = all(len(row) <= max_columns for row in rows)

    return rows

def sort_row(row):
    sorted_row = sorted(row, key=lambda cell: cell[1][3])  # Sort the row based on height
    return sorted_row
def check_h_close(h1, h2, threshold):
    if abs(h1 - h2) < threshold:
        return True
    return False

def split_list_by_h_threshold(lst, threshold):
    result = []
    current_group = [lst[0]]
    for i in range(1, len(lst)):
        prev_tuple = lst[i-1]
        curr_tuple = lst[i]
        if abs(prev_tuple[1][3] - curr_tuple[1][3]) <= threshold:
            current_group.append(curr_tuple)
        else:
            result.append(current_group)
            current_group = [curr_tuple]
    result.append(current_group)
    return result
def new_rows(rows) : 
    new_rowss=[]
    for row in rows :
        sorted_row = sort_row(row)
        new_rowss.append(split_list_by_h_threshold(sorted_row, 25)) 
    return new_rowss
def flatten_rows(new) : 
    full_list = []
    for li in new:
        if len(li) > 1:
            for k in range(len(li)):
                full_list.append(li[k])
        else:
            full_list.extend(li)
    return full_list
def merge_consecutive_rows(rows, threshold):
    merged_rows = []
    current_row = rows[0]

    for i in range(1, len(rows)):
        next_row = rows[i]
        current_y_start, current_h = current_row[-1][1][1], current_row[-1][1][3]
        next_y_start, next_h = next_row[0][1][1], next_row[0][1][3]

        if abs((current_y_start + current_h) - (next_y_start + next_h)) < threshold:
            current_row.extend(next_row)
        else:
            merged_rows.append(current_row)
            current_row = next_row

    merged_rows.append(current_row)
    return merged_rows
def table_rec(columns, rows , cell_data) :
    

    # Create an empty table
    table = []

    # Iterate over each row
    for row in rows:
        table_row = []
        row.sort(key=lambda c: c[0])  # Sort cells within the row based on x-coordinate

        # Iterate over each column
        for col  in columns:
            cell = None

            # Find the cell in the current column that belongs to the current row
            for indic , box  in col:
                if (indic , box) in row:
                    cell=cell_data[indic][-1]

                    break
                    
            # Append the cell to the table row
            table_row.append(cell)

        # Append the table row to the table
        table.append(table_row)

    # Create a DataFrame from the table
    df = pd.DataFrame(table)

    # Set the display options for the DataFrame
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)

    # Print the DataFrame
    tabulated_output = tabulate.tabulate(df, headers='keys', tablefmt='grid')
    return df
