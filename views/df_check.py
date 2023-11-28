from fastapi import FastAPI
import pandas as pd
import numpy as np
def sum_rows(df, row_pairs, result) :
    errors=[]
    for i, (row1, row2) in enumerate(row_pairs):
            selected_rows = df.iloc[[row1, row2]]
            row_sum = selected_rows.sum(axis=0)
            result_row_index = result[i]
            if not row_sum.equals(df.iloc[result_row_index]):
                errors.append({"sum_rows": [row1, row2], "result_row": [result_row_index]}) 
    return errors
        
def convert_to_int(item):
    if item != "null" and item != "":
        return int(item)
    else:
        return 0
def remove_thousands_separator_and_convert(cell):
    if "," in cell:
        return int(cell.replace(",", ""))
    else:
        return cell
    
async def check_dataframe(df , page_number):
    
    columns = df[0]

    df = pd.DataFrame(df[1:], columns=columns)
    df = df.applymap(remove_thousands_separator_and_convert)
    print("corrected data jet" , df , page_number)

    if page_number!=0 :
        column_to_remove = get_column(page_number)
        if column_to_remove in df.columns:
            df = df.drop(column_to_remove, axis=1)
        if "الفقره" in df.columns:
            df = df.drop("الفقره", axis=1)
        df = df.applymap(convert_to_int)

        

    print("data eli jeya " , df)

    errors=[]
    if page_number == 0:
        errors.append({"sum_rows" : [] , "result_row" : []})
        return {"dataframe": df.to_json() ,  "page_number": page_number, "errors": errors}
    if page_number == 1 :
        row_pairs = [(0, 3), (5,7),(9,-10)]
        result= [4,8,11] 
        errors.append(sum_rows(df, row_pairs , result))
    if page_number == 2  or page_number==3:
        row_pairs = [(0, 2)]
        result= [3] 
        errors.append(sum_rows(df, row_pairs , result))
       
        rows_to_sum = [3, 5, 7,9]  
        result_row_index = 10 

        selected_rows = df.iloc[rows_to_sum]
        row_sum = selected_rows.sum(axis=0)

        if not row_sum.equals(df.iloc[result_row_index]):
            errors.append({"sum_rows": [3,5,7,9], "result_row": [10]})
       
    if page_number==4 or page_number == 5 : 
        row_pairs = [(0, 4) , (6,8) , (5,9)]
        result= [5,9,10] 
        errors.append(sum_rows(df, row_pairs , result))

       
    if page_number == 7 or page_number == 6:
        row_pairs = [(0, 7) , (11,12)]
        result= [8,13] 
        errors.append(sum_rows(df, row_pairs , result))

    if page_number == 8  or page_number == 9: 
        row_pairs = [(1,2) , (5,8),(10,11),(13,14)]
        result= [3,9,12 ,15] 
        errors.append(sum_rows(df, row_pairs , result))

    if page_number == 10  or page_number == 11: 
        row_pairs = [(0,1) , (3,4),(8,9),(11,12)]
        result= [2,5,10,13] 
        errors.append(sum_rows(df, row_pairs , result))

    if page_number == 12  or page_number == 13:
        row_pairs = [(0,1) , (4,5),(7,8),(11,12)]
        result= [2,6,9,13] 
        errors.append(sum_rows(df, row_pairs , result))

    if page_number == 14  or page_number == 15: 
        row_pairs = [(0,1)]
        result= [2] 
        errors.append(sum_rows(df, row_pairs , result))
    if page_number == 16 or page_number == 17: 
        row_pairs = [(0,13)]
        result= [14] 
        errors.append(sum_rows(df, row_pairs , result))
       
    if page_number == 18 or page_number == 19: 
        row_pairs = [(0,2) ,(4,6)]
        result= [3 , 7] 
        errors.append(sum_rows(df, row_pairs , result))
    if page_number == 20 or page_number == 21: 
        row_pairs = [(0,5) ,(7,8)]
        result= [6,9] 
        errors.append(sum_rows(df, row_pairs , result))
    if page_number == 22 or page_number == 23: 
        row_pairs = [(0,1)]
        result= [2] 
        errors.append(sum_rows(df, row_pairs , result))
        
    if page_number == 28 or page_number == 29: 
        row_pairs = [(3,5) , (7,8)]
        result= [6,9] 
        errors.append(sum_rows(df, row_pairs , result))
    if page_number == 32 or page_number == 33: 
        row_pairs = [(0,1)]
        result= [2] 
        errors.append(sum_rows(df, row_pairs , result))
       
    if page_number == 34 or page_number == 35: 
        row_pairs = [(0,11) , (0,11)]
        result= [12 , 13] 
        errors.append(sum_rows(df, row_pairs , result))
    if page_number == 36 : 
        row_pairs = [(0,21)]
        result= [22] 
        errors.append(sum_rows(df, row_pairs , result))
    if page_number == 37 : 
        row_pairs = [(0,5)]
        result= [6] 
        errors.append(sum_rows(df, row_pairs , result))
    
    flat_errors = []
    for nested_list in errors:
        if isinstance(nested_list, list):
            flat_errors.extend(nested_list)
        else:
            flat_errors.append(nested_list)
    print("errors are " , errors)
    if page_number !=0:
        new_df = add_details(df , page_number)
    
    new_df = remove_columns_if_exist(new_df)
    if page_number != 0 and page_number!= 1 :
        new_df=add_ref(df, page_number)
    print(new_df.columns)


    
    return {"dataframe": new_df.to_json() , "page_number": page_number, "errors": flat_errors}



def add_details(df , page_number):
    new_col =None
    col_name=None

    if page_number==1 :
        new_col=["نقل الفائض الى" , "مقابيض العنوان الاول", "مقابيض العنوان الثاني", "مقابيض خارج الميزانيه" ,"الجمله", "نفقات العنوان الاول", "نفقات العنوان الثاني", "نفقات خارج الميزانيه","الجمله",
                 "نقل المقابيض" , "نقل المصاريف" , "الفارق الى"]
        
        col_name= 'المقابيض'

    if page_number ==2 or page_number==3:
        
        new_col=["معلوم على العقارات المبنيه" , "معلوم على الاراضي الغير مبنيه","معاليم اخرى", "جمله الصنف الاول" ,"الصنف الثاني", "جمله الصنف الثاني", "الصنف الثالث", "جمله الصنف الثالث",
                    "الصنف الرابع", "جمله الصنف الرابع", "جمله الجزء الاول","مداخيل كراء عقارات معده لنشاط التجاري", "مداخيل كراء عقارات معده لنشاط مهني","ينقل"]
        
        col_name= 'بيان المقابيض'
    if page_number==4 or page_number==5:
        
        new_col=["نقل", "مداخيل كراء عقارات معده لنشاط الصناعي" , "مداخيل كراء عقارات معده للنشاط فليحي", "مداخيل كراء عقارات معده للسكن","موارد اخرى",
                    "جمله الصنف الخامس" , "المناب من المال المشترك","موارد منقوله من فوائد العنوان الاول","موارد اخرى", "جمله الصنف السادس", "جمله الجزء الثاني",
                    "جمله موارد العنوان الاول"]
        col_name='بيان المقابيض'
    if page_number==6 or page_number==7 :
        new_col=["نقل فواضل", "موارد السنه" , "نقل فواضل", "موارد السنه" , "نقل فواضل", "موارد السنه", "نقل فواضل", "موارد السنه", "جمله الفصل 01-70", "نقل فواضل", "موارد السنه", "ينقل"]
        col_name='بيان المقابيض'


    if page_number==8 or page_number==9 :
        new_col=["نقل" ,  "نقل فواضل", "موارد السنه" , "جمله الفصل 02-70" , "جمله الصنف الرابع" ,"نقل فواضل", "المبالغ المقامه من الفوائض غير المستعمله من العنوان الاول للسنه الاخيره",
                    "الفوائض غير المستعمله من العنوان الاول للسنه السابقه للسنه الاخيره والمؤمنه", "بالعمليات الخارجه على الميزانيه", "جمله الفصل 01-80" , "نقل فواضل", "موارد السنه" , "جمله الفصل 02-80", "نقل فواضل", "موارد السنه",  "جمله الفصل 03-80"]
        col_name='بيان المقابيض'

    if page_number == 10 or page_number==11:
        new_col=[ "نقل فواضل", "موارد السنه" ,  "جمله الفصل 04-80" ,  "نقل فواضل", "موارد السنه" ,  "جمله الفصل 05-80", "جمله الصنف الثامن" , "جمله الجزء الثالث" ,"نقل فواضل", "موارد السنه" ,  "جمله الفصل 01-90",  "نقل فواضل", "موارد السنه" ,  "جمله الفصل 02-90"]
        
        col_name='بيان المقابيض'



    if page_number==12 or page_number==13:
        new_col=[ "نقل فواضل", "موارد السنه" ,  "جمله الفصل 03-90" , "جمله الصنف التاسع",  "نقل فواضل", "موارد السنه" ,  "جمله الفصل 01-100" ,"نقل فواضل", "موارد السنه" ,  "جمله الفصل 02-100", "جمله الصنف العاشر", "نقل فواضل", "موارد السنه" ,  "جمله الفصل 01-110"]
        col_name='بيان المقابيض'


    if page_number== 14 or page_number==15:
        new_col=[ "نقل فواضل", "موارد السنه" ,  "جمله الفصل 02-110" , "جمله الصنف الحادي عشر" , "جمله الجزء الرابع"]
        col_name='بيان المقابيض'



    if page_number == 16 or page_number==17:
        new_col =["الباب الثاني رئاسه الجمهوريه", "الباب الخامس شؤون المراه والاسره", "الباب السابع وزاره الداخليه", "الباب الثالث عشر الماليه" , 
                    "الباب العشرون التجهيز" , "الباب الحادي والعشرون البيئه" , "الباب الثالث والعشرون السياحه" , "الباب الخامس والعشرون الثقافه" , "الباب السادس والعشرون الرياضه والتربيه البدنيه" , "الباب الثامن والعشرون الشؤون الاجتماعي" , "الباب الخامس والثلاثون الطفوله" ,
                    "الباب السادس والثلاثون الشباب" ,"الباب التسعون مساهمات ماليه مختلفه لانجاز مشاريع ذات صبغه محليه" , "جمله الصنف الثاني عشر", "جمله الجزء الخامس", "جمله موارد العنوان الثاني", "مجموع موارد ميزانيه البلديه "]
        col_name='بيان المقابيض'

    if page_number==18 or page_number==19:
        new_col=["جمله الفصل 100-01", "جمله الفصل 101-01", "جمله الفصل 102-01" , "جمله القسم الاول" ,"جمله الفصل 201-02", "جمله الفصل 202-02" , "جمله الفصل 203-02" , "جمله القسم الثاني"]
        col_name='بيان النفقات'

    if page_number==20 or page_number==21:
        new_col=["جمله الفصل 302-03", "جمله الفصل 303-03", "جمله الفصل 305-03", "جمله الفصل 306-03", "جمله الفصل 307-03","جمله الفصل 310-03" ,"جمله القسم الثالث","جمله الفصل 400-04", "جمله الفصل 401-04" , "جمله القسم الرابع" , "جمله الجزء الاول"]
        col_name='بيان النفقات'
    if page_number==22 or page_number==23:
        new_col=["جمله الفصل 500-05", "جمله الفصل 501-05" , "جمله القسم الخامس", "جمله الجزء الثاني" ,"جمله نفقات العنوان الاول", "المصاريف الماذونه بعنوان الفوائض" , "الجمله العامه لنفقات العنوان الاول"]
        col_name='بيان النفقات'
    if page_number==24 or page_number==25:
        new_col=["جمله الفصل 600-06", "جمله الفصل 601-06","جمله الفصل 602-06" , "جمله الفصل 603-06","جمله الفصل 604-06","جمله الفصل 605-06","جمله الفصل 606-06","جمله الفصل 607-06","جمله الفصل 608-06","جمله الفصل 609-06"]
        col_name='بيان النفقات'
    if page_number==26 or page_number==27:
        new_col=["جمله الفصل 610-06" , "جمله الفصل 611-06" , "جمله الفصل 612-06", "جمله الفصل 613-06","جمله الفصل 614-06","جمله الفصل 615-06"]
        col_name='بيان النفقات'
    if page_number==28 or page_number==29:
        new_col=["جمله الفصل 616-06", "جمله الفصل 617-06", "جمله القسم السادس","جمله الفصل 810-07", "جمله الفصل 811-07" , "جمله الفصل 827-07", "جمله القسم السابع", "جمله الفصل 900-08", "جمله الفصل 901-08" , "جمله القسم الثامن"]
        col_name='بيان النفقات'
    if page_number==30 or page_number==31:
        new_col=["يتم بهذا القسم عند الاقتضاء ادراج نفس الفصول المضمنه بالقسمين السادس والسابع مع تغيير الرقم المميز للقسم التاسع بدل من السادس او السابع" ,
                    "جمله القسم التاسع" , "جمله الجزء الثالث"]
        col_name='بيان النفقات'
    if page_number==32 or page_number==33:
        new_col=["جمله الفصل 950-10", "جمله الفصل 951-10" , "جمله القسم العاشر" ,"جمله الجزء الرابع" , "جمله نفقات الجزئين الثالث والرابع", "المصاريف الماذونه بعنوان الفوائض" , "الجمله العامه لنفقات الجزئين 3 و 4"]
        col_name='بيان النفقات'
    if page_number==34 or page_number==35:
        new_col=["الباب الثاني رئاسه الجمهوريه", "الباب الخامس شؤون المراه والاسره", "الباب السابع وزاره الداخليه", "الباب الثالث عشر الماليه" , 
                    "الباب العشرون التجهيز" , "الباب الحادي والعشرون البيئه" , "الباب الثالث والعشرون السياحه" , "الباب الخامس والعشرون الثقافه" , "الباب السادس والعشرون الرياضه والتربيه البدنيه" , "الباب الثامن والعشرون الشؤون الاجتماعي" , "الباب الخامس والثلاثون الطفوله" ,
                    "الباب السادس والثلاثون الشباب" ,"الباب 90 مساهمات ماليه مختلفه لانجاز مشاريع ضد صبغه محليه" , "جمله القسم الحادي عشر", "جمله الجزء الخامس", "مصاريف الماذونه بعنوان الفوائض", "جمله العامه للنفقات الجزء الخامس",
                    "الجمله العامه لنفقات العنوان الثاني", "الجمله العامه لنفقات ميزانيه البلديه"]
        col_name ='بيان النفقات'
    if page_number==36 :
        new_col=["ايداعات مختلفه", "مقابيض مستخلصه قبل اعداد اذان استخلاص", "مخاصيم", "ضمانات", "رفض حساب جاري بريدي بعد التحويل", "بقايا للخلاص نقدا", "بعنوان سنه", "بعنوان سنه", "بعنوان سنه", "بعنوان سنه", "بقايا للخلاص بالتحويلات", "الصندوق الوطني لتحسين السكن", "اذاعات مقابل تسبيقات على الحساب لتسديد صفقات مختلفه","وكاله التهذيب والتجديد العمراني",
                    "مؤسسات اخرى", "المال الاحتياطي", "المال الاحتياطي من العنوان الاول %20","المال الاحتياطي من العنوان الاول %80", "المال الاحتياطي من العنوان الثاني الجزئين ثلاثه واربعه", "حساب التحويل العنوان الثاني الجزء الخامس", "اجور غير خالصه", "زياده %5 في مصاريف التتبع", "مجموع الايداعات والتامينات"]
        col_name = 'بيان الحسابات'
    if page_number==37 :
        new_col=["تسبقات مرخص فيها" , "تسديد مبالغ بعنوان صفقات مختلفه مبرمه مع مؤسسات", "وكاله التهذيب والتجديد العمراني", "مؤسسات اخرى", "تسبقات للوكلاء" , "تسبقة بعنوان رفض وثائق صرف" ,"تسبقة لتسوية عجز " , "مجموع التسبقات" , "الجزء الأول :الإيداعات والتأمينات", "الجزء الثاني :التسبقات"]
        col_name = 'بيان الحسابات'
    num_nulls_needed = len(df) - len(new_col)

    if num_nulls_needed > 0:
        new_col.extend([np.nan] * num_nulls_needed)
    elif num_nulls_needed < 0:
        # Calculate the number of rows to add
        num_rows_to_add = abs(num_nulls_needed)

        # Create a DataFrame with the same columns as df
        additional_rows = pd.DataFrame(columns=df.columns, data=[[np.nan] * len(df.columns)] * num_rows_to_add)

        # Concatenate the additional rows to the original DataFrame
        df = pd.concat([df, additional_rows], ignore_index=True)

        # Ensure that new_col and df have the same length
        new_col = new_col[:len(df)]
    df.insert(df.shape[1],col_name, new_col)


    return df
def get_column(page_number) :
   
    if page_number == 1 :
        column_to_remove = "المقابيض"
    if 2 <= page_number <= 17:
        column_to_remove= 'بيان المقابيض'
    if 18 <= page_number <= 35  :
        column_to_remove = 'بيان النفقات'
    if 36 <= page_number <= 37:
        column_to_remove='بيان الحسابات'
    return column_to_remove
def add_columns(df, page_number):
    if page_number == 0:
        print("data page 0 " , df)
        headers = [
            ["المبلغ الجملي لمقابيض الميزانيه","", "المبلغ الجمله لمصاريف الميزانيه", "نتيجه الجمله الفائض",  "نتيجه الجمله العجز"],
            ["المقابيض", "المصاريف", "المقابيض المستعمله لتسديد مصاريف بالجزءين 3 و4 من العنوان الثاني", "النتيجه الفائض", "النتيجه العجز"],
            ["المقابيض", "المصاريف", "المصاريف المسدده بالجزئين 3 و4 من العنوان الثاني بموارد من العنوان الاول", "النتيجه الفائض", "النتيجه العجز"],
            ["المقابيض", "","المصاريف", "النتيجه الفائض","النتيجه العجز"]
        ]
        print(f"Header Row Type: {type(headers[0])}")

        result_df = pd.DataFrame()
        headers = [sublist[::-1] for sublist in headers]

        # Iterate through the rows and headers, and add them to the result DataFrame
        for data_row, header_row in zip(df.values, headers):
            # Create DataFrames for headers and data
            header_df = pd.DataFrame([header_row], columns=df.columns)
            data_df = pd.DataFrame([data_row], columns=df.columns)

            # Concatenate the header row and data row vertically
            table_df = pd.concat([header_df, data_df], ignore_index=True)

            # Append the table to the result DataFrame
            result_df = pd.concat([result_df, table_df], ignore_index=True)

        print(result_df)
        return result_df


    if page_number==2 or page_number==4 : 
        col=["الفصل" ,  "الفقره", "تقديرات الميزانيه" , "التنقيحات الحاصله بقرارات بالنقص" , "التنقيحات الحاصله بقرارات بالزياده", "التقديرات النهائيه"]

    if page_number in [3,5  , 7, 9 ,11 ,13, 15, 17 ]: 
        col=["بقايا الاستخلاص الى 31-12" , "تثقليات عن طريق اذون وقتيه", "تثقيلات عن طريق اذون نهائيه" , "المجموع" , "المطروحات", "المبالغ الواجب استخلاصها",
             "المقابيض المنجزه عن طريق اذون وقتيه" , "المقابيل المنجزه عن طريق اذون نهائيه", "المقابيض المنجزه المجموع", "بقايا الاستخلاص الى 31-12 (2)-(1) "]   

    if page_number in [ 6 , 8, 10, 12 ,14] : 
        col=["الفصل" ,  "الفقره", "الفقره الفرعيه", "تقديرات الميزانيه" , "التنقيحات الحاصله بقرارات بالنقص" , "التنقيحات الحاصله بقرارات بالزياده", "التقديرات النهائيه"]
    if page_number == 16 : 
        col=[ "تقديرات الميزانيه" , "التنقيحات الحاصله بقرارات بالنقص" , "التنقيحات الحاصله بقرارات بالزياده", "التقديرات النهائيه"]
    if page_number in [18  , 20 ,22 , 24 ,26,28 ,30 ,32]:
        col =["الفصل" ,  "الفقره", "الفقره الفرعيه" ,"الاعتمادات المرسمه بالميزانيه" , "التنقيحات الحاصله بقرارات بالنقص", "التنقيحات الحاصله بقرارات بالزياده", "مجموع الاعتمادات"]

    if page_number==34 :
        col =[  "الاعتمادات المرسمه بالميزانيه" , "التنقيحات الحاصله بقرارات بالنقص", "التنقيحات الحاصله بقرارات بالزياده", "مجموع الاعتمادات"]

    if page_number==36 :
        col=["الحساب" , "الحساب الفرعي",  "بقايا الايداعات في 31 ديسمبر", "مقابيض المنجزه اثناء السنه","مجموع المقابيض","المصاريف المنجزه اثناء السنه","بقايا الايداعات الى 31 ديسمبر"]
    if page_number in [19,21 , 23] :
        col=["تحويل الاعتمادات بالنقص", "تحويل الاعتمادات بالزياده", "الاعتمادات النهائيه","مصاريف ماموره","اعتمادات غير مستعمله"]  
    if page_number in [25,27 , 29 ,31,33,35]:
        col=["تحويل الاعتمادات بالنقص", "تحويل الاعتمادات بالزياده", "الاعتمادات النهائيه","مصاريف ماموره","اعتمادات غير مستعمله" , "عمودان مخصصان للاذن بالدفع الاعتمادات التي تنقل", "عمودان مخصصان للاذن بالدفع الاعتمادات التي تلغى نهائيا"]  
    if page_number == 37 : 
        col=["الحساب" , "الحساب الفرعي" , "البقايا الحاصله الى 31 ديسمبر" , "المقابيض الحاصله خلال السنه" , "بقايا تطبيقات لاخر يوم من السنه الفارطه","جمله المقابيض" , "تطبيقات المنجزه اثناء السنه", "مصاريف المنجزه اثناء السنه" , "جمله التسبيقات" ,"مقابيض المنجزه اثناء السنه" , "بقايا الاداعات والتامينات" ,"بقايا التسبيقات الى 31 ديسمبر"]
    
    col.reverse()

    if len(col) < len(df.columns):
        # Assign values from cols to existing columns
        df.columns = col + list(df.columns[len(col):])
    elif len(col) > len(df.columns):
        # Add null columns to df to match the length of cols
        df = pd.concat([df, pd.DataFrame(columns=col[len(df.columns):])], axis=1)
    else:
        # If lengths are equal, just assign the columns
        df.columns = col    
    return df
def remove_columns_if_exist(df):
    columns_to_remove= ["الفصل", "الفقره", "الفقره الفرعيه", "الحساب" , "الحساب الفرعي"]
    existing_columns = df.columns.tolist()
    columns_to_drop = [col for col in columns_to_remove if col in existing_columns]
    df = df.drop(columns=columns_to_drop, axis=1)
    return df



def add_ref(df ,page_number) :
    new_col =None
    
    if page_number == 2 or page_number==3 :
        
        new_col =["11-01", "11-02", "","","","","","","","","" ,"52-01","52-02",""]
    if page_number == 4 or page_number==5 :
        new_col=["","52-03","52-04" ,"52-05","","","60-01" ,"60-02","","","",""]
    if page_number ==6 or page_number==7 : 
        new_col =["70-01-01-001","70-01-01-002","70-01-02-001","70-01-02-002","70-01-03-001","70-01-03-002","70-01-04-001","70-01-04-002","","70-02-01-001",
                  "70-02-01-002",""]
    if page_number==8 or page_number==9:
        new_col=["","70-02-02-001","70-02-02-002","","","80-01-01","80-01-02","80-01-03","","80-02-01","80-02-02","", "80-03-01","80-03-02",""]
    if page_number==10 or page_number==11:
        new_col=["80-04-01","80-04-02","","","80-05-01","80-05-02","","","","90-01-01","90-01-02","" ,"90-02-01","90-02-02",""]
    if page_number==12 or page_number==13:
        new_col=["90-03-01","90-03-02","","","100-01-01","100-01-02","","100-02-01","100-02-02","","","110-01-01","110-01-02" ,""]
    if page_number==14 or page_number==15 :
        new_col=["110-02-01", "110-02-02","","","",""]
    if page_number==18 or page_number==19:
        new_col=["01.100" ,"01.101","01.102","","02.201","02.202","02.230",""]
    if page_number==20 or page_number==21:
        new_col=["03.302","03.303","03.305","03.306","03.307","03.310","","04.400","04.401","",""]

    if page_number==22 or page_number==23 :
        new_col=["05.500", "05.501","","","","",""]
    if page_number==24 or page_number==25 :
        new_col=["06.600","06.601","06.602","06.603","06.604","06.605","06.606","06.607","06.608","06.609"]

    if page_number==26 or page_number==27 :
        new_col=["06.610", "06.611","06.612","06.613","06.614","06.615"]
    if page_number==28 or page_number==29 :
        new_col=["06.616", "06.617","","07.810","07.811","07.827","","08.900","08.901",""]
    if page_number==30 or page_number==31 :
        new_col=["","","",""]
    if page_number==32 or page_number==33 :
        new_col=["10.950", "10.951","","","","",""]
    if page_number==36 :
        new_col=["1010","1020","1030","1040","1050","1060","","","","","1070","1080","1090","1090-01","1090-99","1100","1100-01","1100-02","1100-03","1110","1120","1130",""]
    if page_number==37 :
        new_col=["2010" , "2020","2020-01","2020-99","2030","2040","2050",""]
    
    num_nulls_needed = len(df) - len(new_col)

    if num_nulls_needed > 0:
        new_col.extend([np.nan] * num_nulls_needed)
    else:
        # Calculate the number of new rows to add
        num_new_rows = abs(num_nulls_needed)
        # Create a DataFrame with new rows and append it to the original DataFrame
        new_rows = pd.DataFrame(np.nan, index=range(num_new_rows), columns=df.columns)
        df = pd.concat([df, new_rows], ignore_index=True)

    


    df.insert(df.shape[1], "الفقره", new_col)

    return df



