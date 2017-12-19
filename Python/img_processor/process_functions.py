import cv2

def process(filename, key):
    "Convert img file into HSV(Hue Saturation Value) matrix"
    try:
        # Read in File
        img = cv2.imread(filename)
        # Resize Image
        resized_img = cv2.resize(img,(10,10))
        # Convert RBG to Hue Saturation Value Matrix
        hsv_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
        return hsv_img
    except:
        raise ('ERROR: ' + filename)

def hue_extract(hsv_matrix):
    """Extract just Hue from the HSV Matrix"""
    hue_only_matrix = []
    for row in hsv_matrix:
        hue_only_row = []
        for pixel in row:
            hue_only_row.append(int(pixel[0]))
        hue_only_matrix.append(hue_only_row)
    return hue_only_matrix
