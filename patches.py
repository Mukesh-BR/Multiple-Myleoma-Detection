import cv2
import os

input_path_img="F:\\MiMM_SBILab\\train\\images"
input_path_mask="F:\\MiMM_SBILab\\train\\masks"

output_path_img="F:\\MiMM_SBILab\\patches\\train\\images\\0"
output_path_mask="F:\\MiMM_SBILab\\patches\\train\\masks\\0"

def make_patch(filename,img_path,mask_path,output_path_img,output_path_mask,patch_size):
    img=cv2.imread(img_path)
    mask=cv2.imread(mask_path)
    print(img.shape)
    print(mask.shape)
    i=0
    filename=filename.split(".")[0]
    for y in range(0,img.shape[0]-patch_size,patch_size):
        for x in range(0,img.shape[1],patch_size):
            crop_img = img[y:y+patch_size, x:x+patch_size]
            crop_mask = mask[y:y+patch_size,x:x+patch_size]
            str_name=filename+str(i)+".jpg"
            status1=cv2.imwrite(output_path_img+"\\"+str_name, crop_img)
            status2=cv2.imwrite(output_path_mask+"\\"+str_name, crop_mask)
            print(status1,status2,output_path_img+"\\"+str_name,crop_img.shape)
            i+=1
    
    y=img.shape[0]-patch_size
    for x in range(0,img.shape[1],patch_size):
            crop_img = img[y:y+patch_size, x:x+patch_size]
            crop_mask = mask[y:y+patch_size,x:x+patch_size]
            str_name=filename+str(i)+".jpg"
            status1=cv2.imwrite(output_path_img+"\\"+str_name, crop_img)
            status2=cv2.imwrite(output_path_mask+"\\"+str_name, crop_mask)
            print(status1,status2,output_path_img+"\\"+str_name,crop_img.shape)
            i+=1




patch_size=512



for root, dirs, files in os.walk(input_path_img):
    for filename in files:
        filename=filename.split(".")[0]
        img=input_path_img+"\\"+filename+".bmp"
        mask=input_path_mask+"\\"+filename+"_mask_converted.bmp"
        make_patch(filename,img,mask,output_path_img,output_path_mask,patch_size)
# crop_img = img[y:y+h, x:x+w]
# cv2.imwrite(str(y)+"_"+str(x)+".png", crop_img)
# print("style"+str(y)+"_"+str(x)+".png")
