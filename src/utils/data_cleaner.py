import os
import patoolib
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import shutil

# extract coloured images
def extract_coloured(folder):
    zip_files = os.listdir(folder)
    output_folder = r'C:\Users\safwa\Desktop\dataset\coloured'
    for i, filename in enumerate(zip_files):
        filepath = os.path.join(folder, filename)
        patoolib.extract_archive(filepath, outdir=output_folder)

# extract uncoloured images
def extract_uncoloured(folder):
    zip_files = sorted(os.listdir(folder), key=lambda x: int(x.split()[2]))
    output_folder = r'C:\Users\safwa\Desktop\dataset\uncoloured'
    for i, filename in enumerate(zip_files):
        filepath = os.path.join(folder, filename)
        subfolder = os.path.join(output_folder, "uncoloured_"+str(i+1))
        os.mkdir(subfolder)
        patoolib.extract_archive(filepath, outdir=subfolder)


def build_dataset(dataset_folder, coloured_folder, uncoloured_folder):
    dataset_coloured_folder = os.path.join(dataset_folder, "coloured")
    dataset_uncoloured_folder = os.path.join(dataset_folder, "uncoloured")
    filtered_out_log = r'C:\Users\safwa\Desktop\anomalies\filtered.txt'
    THRESHOLD = 4500

    variances = []

    if os.path.exists(filtered_out_log):
        os.remove(filtered_out_log)

    for i, folder in enumerate(os.listdir(coloured_folder)):
        iid = 0
        a_iid = 0
        print(f"processing folder {folder}...")
        subfolder = os.path.join(coloured_folder, folder)
        for j, file in enumerate(os.listdir(subfolder)):
            coloured_file = os.path.join(subfolder, file)

            uncoloured_subfolder = os.path.join(uncoloured_folder, os.listdir(uncoloured_folder)[i])
            uncoloured_file = os.path.join(uncoloured_subfolder, os.listdir(uncoloured_subfolder)[j])
            
            coloured_image = cv.imread(coloured_file)
            gray = cv.cvtColor(coloured_image, cv.COLOR_BGR2GRAY)
            var = np.var(gray)
            # variances.append(var)

            # resize
            uncoloured_image = cv.imread(uncoloured_file)
            coloured_image = cv.resize(coloured_image, (256, 256), interpolation=cv.INTER_AREA)
            uncoloured_image = cv.resize(uncoloured_image, (256, 256), interpolation=cv.INTER_AREA)

            # remove low variance images
            if (var < THRESHOLD): 
                with open(filtered_out_log, 'a') as f:
                    f.write(f"{coloured_file}: {var}\n")
                cv.imwrite(os.path.join(r'C:\Users\safwa\Desktop\anomalies\coloured', f'{i:02}_{a_iid:03}.jpg'), coloured_image)
                cv.imwrite(os.path.join(r'C:\Users\safwa\Desktop\anomalies\uncoloured', f'{i:02}_{a_iid:03}.jpg'), uncoloured_image)
                a_iid += 1
            else:                
                cv.imwrite(os.path.join(dataset_coloured_folder, f'{i:02}_{iid:03}.jpg'), coloured_image)
                cv.imwrite(os.path.join(dataset_uncoloured_folder, f'{i:02}_{iid:03}.jpg'), uncoloured_image)
                iid += 1
            
        # plt.figure()
        # plt.xlabel("grayscale variance")
        # plt.scatter(variances, [0]*len(variances))
        # plt.show()
        # break

def main():
    # extract_coloured(r'C:\Users\safwa\Desktop\coloured_zips')
    # extract_uncoloured(r'C:\Users\safwa\Desktop\uncoloured_zips')
    
    # build_dataset(r'C:\Users\safwa\Desktop\dataset', r'C:\Users\safwa\Desktop\coloured', r'C:\Users\safwa\Desktop\uncoloured')

    u_image = cv.imread(r"C:\Users\safwa\Desktop\uncoloured\uncoloured_01\ONE PIECE 1 - p002 [aKraa].jpg")
    h, w, _ = u_image.shape
    resized = cv.resize(u_image, (round(w*1.8), round(h*1.8)))
    
    c_image = cv.imread(r"C:\Users\safwa\Desktop\coloured\coloured_01\002.jpg")
    ch, cw, _ = c_image.shape

    cropped = c_image[0:ch, (cw-round(w*1.8)):cw]

    print(resized.shape)
    print(cropped.shape)
    cv.imwrite('u_resized.jpg', resized)
    cv.imwrite('c_cropped.jpg', cropped)

if __name__ == "__main__":
    main()