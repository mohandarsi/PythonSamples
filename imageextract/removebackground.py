import glob
import os
import cv2
import argparse
from pathlib import Path 
from PIL import Image
from natsort import natsorted 

debug = False
debugFolder = None
pdf_name = "generated.pdf" 

def get_images_by_ext(directory:str,ext:str):
    return glob.glob( os.path.join(directory,'*.'+ext) ) # find all pngs with one level depth

def remove_back_ground(img,filepath:str): #Mat

    print("Processing image {0}".format(filepath))
    # convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find Canny edges
    edged = cv2.Canny(gray, 30, 200)
   
    # Finding Contours
    # Use a copy of the image e.g. edged.copy()
    # since findContours alters the image
    contours, hierarchy = cv2.findContours(edged, 
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    #No contours
    if( not bool(contours)):
       return img
    
    big_contour = max(contours, key=cv2.contourArea)

    # get bounding boxes
    x,y,w,h = cv2.boundingRect(big_contour)
    
    # 1907 X 1351
    if(w > 1800 and h >1200):
        crop = img[y:y+h, x:x+w]
        return crop
    
    if debug :
        cv2.imwrite(os.path.join(debugFolder,Path(filepath).stem +'_canny.png'), edged)
        clone = img.copy()
        cv2.drawContours(clone, contours, -1, (0, 255, 0), 3)
        cv2.imwrite(os.path.join(debugFolder,Path(filepath).stem +'_contour.png'), clone)

    return None


def generate_pdf_from_images(dirpath:str,ext:str): 

    img_filenames = get_images_by_ext(dirpath,ext)

    natsort_file_names = natsorted(img_filenames)
    
    image_list =[]
    for filepath in natsort_file_names:
        print("Adding image {0} to pdf".format(Path(filepath).stem))
        image = Image.open(filepath).convert('RGB')
        image_list.append(image)
    
   
    image_list[0].save(os.path.join(dirpath,pdf_name), save_all=True, append_images=image_list[1:])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extracts foreground image from backgorund') 
    parser.add_argument('-i','--input', help='source directrory where full image(s) are present', type=str,required=True)
    parser.add_argument('-o','--output', help='destination directory to store processed images', type=str,required=True)
    parser.add_argument('-e','--imageext', help='Images types to process png or jpg', type=str,required=True)

    args = parser.parse_args()

    img_filenames = get_images_by_ext(args.input,args.imageext)
    debugFolder = args.output

    imgShape = (1907,1351)
    for filepath in img_filenames:
        # read image
        img = cv2.imread(filepath)
        # remove background from the image
        crop = remove_back_ground(img,filepath)
        if (crop is None):
            crop = cv2.resize(img, (1907,1351), interpolation = cv2.INTER_AREA)

        cv2.imwrite(os.path.join(args.output,Path(filepath).name), crop)

    generate_pdf_from_images(args.output,args.imageext)
