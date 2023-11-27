# Financial OCR App

## Description
This project aims to create an OCR-based application for extracting financial data from PDF documents. The application undergoes a series of steps, including document upload, data preprocessing, OCR, table reconstruction, error detection, verification, and storage in MongoDB.

## Features
- Document upload and preprocessing
- Table structure detection and cell identification
- OCR for numeric data (trocr) and Arabic text in table headers (easyocr)
- Table reconstruction based on detected cells
- Error detection for sum and difference mismatches
- User and system verification of financial data
- Storage of verified data in MongoDB

## Installation
1. Clone the repository: `git clone https://github.com/ArifMariem/financial-ocr-app.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Set up MongoDB and configure connection settings in `config.py`.

## Usage
1. Upload a PDF document containing financial data.
2. The application detects the table structure and identifies cells.
3. Numeric data is OCR processed using trocr, and Arabic text in table headers is processed using easyocr.
4. Reconstructed tables are generated based on the detected cells.
5. The application checks for errors in the financial data, ensuring the sum and difference of rows match the total row.
6. Users verify the correctness of the data both manually and through system checks.
7. Verified financial data is stored in MongoDB for future reference.

## Data Preprocessing
The preprocessing phase for table recognition and detection encompasses the following key steps: 
1. Document Upload: Users initiate the process by uploading the document containing tabular data.
2. Image Compression: To optimize processing speed and resource utilization, the uploaded images undergo compression without compromising data integrity.
3. Gray Scale Conversion: The images are converted to grayscale to simplify subsequent image processing operations.
4. Binarization: Utilizing thresholding techniques, the grayscale images are transformed into binary images, enhancing the contrast between foreground and background.
5. Image Inversion: Inverting the binary images ensures that table structures are appropriately highlighted for subsequent detection processes.
6. Horizontal and Vertical Lines Detection: Employing algorithms for line detection, the system identifies both horizontal and vertical lines within the document, aiding in table structure recognition.
7. Lines Intersection Detection: Intersection points of detected lines are identified, contributing to the accurate delineation of table cells.
8. Contours Detection: Contours are extracted from the binary images, outlining distinct shapes and structures present in the document.
9. Cell Segmentation: Based on the identified contours, the system segments the document into individual cells, laying the groundwork for subsequent optical character recognition (OCR) processes.
The preprocessing phase for cells recognition and detection encompasses the following key steps:
1. Binarization
2. Top Hat Transformation: Applying the top-hat transformation helps in highlighting subtle details and fine structures within the document. This is particularly useful for improving the visibility of smaller elements, such as text and lines.
3. Text Region Detection: The system identifies regions within the document that contain text with MSER tools. This step aims to isolate areas where cells and textual information coexist.
4. Region Filtering: The detected regions are filtered based on predefined criteria to focus on areas likely to contain cells. This helps eliminate unnecessary noise and ensures that only relevant portions of the document are considered for further analysis.
5. Empty Cells Regions: Special attention is to regions that appear to be empty cells. By distinguishing between regions containing text and those that are seemingly empty, the system improves its ability to accurately recognize and segment cells.
6. Dotted Line Detection: Implement algorithms to specifically detect dotted lines within the cells. This step is crucial for recognizing boundaries or divisions within the cells that may be represented by dotted lines, enhancing the precision of cell segmentation.

## Text detection : OCR
The Optical Character Recognition (OCR) phase involves the extraction of textual information from the preprocessed document, with a tailored approach to handle both numerical strings and Arabic text in table headers.

1. Fine-Tuning of trocr: The trocr OCR model undergoes a fine-tuning process specific to the characteristics of financial data. This step optimizes the model for accurately detecting numerical strings that represent financial information within the tables.
2. Numerical String Detection: Leveraging the fine-tuned trocr model, the system identifies and extracts numerical strings containing crucial financial data. This process ensures precise recognition of numeric values within the document.
3. EasyOCR for Arabic Text Detection in Headers: The easyocr module is specifically applied to detect and extract Arabic text within the headers of tables. This ensures accurate recognition of textual information in the Arabic language, contributing to the comprehensive extraction of financial data.
## Table Reconstruction
Detected cells are organized to reconstruct tables, providing a clear representation of the financial data.

## Error Detection and Verification
The application identifies errors in the financial data, such as discrepancies in row sums and differences. Users and system checks are performed to ensure data accuracy.

## Data Storage
Verified financial data is stored in MongoDB, facilitating easy access and retrieval.

## Dependencies
- Python 3.9.13
- trocr
- easyocr
- MongoDB
- OpenCV


## Contact Information
For questions or support, contact us at mariemarif98@gmail.com.

## Acknowledgements
- Special thanks to the developers of trocr and easyocr for their valuable OCR libraries.
