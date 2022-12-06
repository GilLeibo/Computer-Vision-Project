# Python program to convert
# numpy array to image

# import required libraries
import numpy as np
from PIL  import Image as im
import subprocess


# define a main function
def generatePics(i):
    # create a numpy array from scratch
    # using arange function.
    # 1440x1080 = 1555200 is the amount
    # of pixels.
    # np.uint8 is a data type containing
    # numbers ranging from 0 to 255
    # and no non-negative integers
    array= np.random.randint(0,256, size=1555200, dtype= np.uint8 )

    # Reshape the array into a
    # familiar resoluition
    array = np.reshape(array, (1080, 1440))

    # creating image object of
    # above array
    data = im.fromarray(array)

    # saving the final output
    # as a PNG file
    data.save('Random_Pictures/lindau_00000'+ str(i) +'_000019_leftImg8bit' + '.png')


# driver code
if __name__ == "__main__":
    pictures_num = input("Enter number of pictures to generate: \n")
    cmd = 'mkdir -p Random_Pictures'
    subprocess.run(cmd, shell=True)
    for i in range (int(pictures_num)):
        generatePics(i)
    print("done! Generated " + pictures_num + " pictures")
