from PIL import Image

import os,glob

def batch_image(dir_input,dir_output):
    if not os.path.exists(dir_output):
        print(dir_output,'is not existed')
        os.mkdir(dir_output)
    else:
        count = 0
        for files in glob.glob(dir_input+'/*'):
            filepath,filename = os.path.split(files)
            file_output = filename[0:10] + '.jpg'
            try:
                im = Image.open(files)
                new_path = os.path.join(dir_output, file_output)
                im.save(new_path)
            except Exception as e:
                print()
            count+=1


if __name__ == '__main__':
    batch_image('C:\\Users\\57261\\Desktop\\BioId','C:\\Users\\57261\\Desktop\\bioi6d')
