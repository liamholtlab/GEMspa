
from skimage import io
from nd2reader import ND2Reader

dir_ = "/Users/sarahkeegan/Dropbox/mac_files/holtlab/data_and_results/GEMs/Liam"
file_1="YM201636-500nM-20H001.nd2"
file_2="YM201636-500nM-20H002.nd2"

#file_1 is the gems
#file_2 is the nuclei and the bright field

def save_metadata(metadata, file_name):
    f = open(file_name, 'w')
    for md in metadata.keys():
        f.write(md + ": " + str(metadata[md]) + '\n')
    f.close()

def read_cell_image(input_file):
    with ND2Reader(input_file) as images:
        save_metadata(images.metadata, input_file[:-4] + '_metadata.txt')
        num_frames = images.metadata['num_frames']
        if(num_frames > 1):
            print("Error 1")
            return 1
        num_ch = len(images.metadata['channels'])
        if (num_ch == 2):
            print("Error 1")
            return 1
        else:
            color_img = False

def read_gems_movie(input_file):

    with ND2Reader(input_file) as images:
        save_metadata(images.metadata, input_file[:-4] + '_metadata.txt')
        num_frames = images.metadata['num_frames']

        duration = images.metadata['experiment']['loops'][0]['duration']
        frame_rate = (duration / 1000) / (num_frames - 1)


        num_ch = len(images.metadata['channels'])
        if (num_ch > 1):
            color_img = True
        else:
            color_img = False

        if (color_img):
            images.bundle_axes = 'yxc'
        else:
            images.bundle_axes = 'yx'

        try:
            images.iter_axes = 'z'
        except ValueError:
            pass
        return images


gems_images = read_gems_movie(dir_ + '/' + file_1)
color_image = read_cell_image(dir_ + '/' + file_2)




