from tifffile import imread, imwrite
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

my_folder='/Users/snk218/Dropbox/Mac/Desktop/plain_tiffs'
my_output_folder='/Users/snk218/Dropbox/Mac/Desktop/fixed_tiffs'

N_c = 4
N_z = 11
N_t = 90
h = 2048
w = 2048

file_pre = "infection_time_90min_1_MMStack_Pos0"
file_suf = ".ome.tif"

# infection_time_90min_1_MMStack_Pos0_1.ome
MAX_ID = 2
id_list = list(range(1, MAX_ID + 1))
id_list = ['_' + str(x) for x in id_list]
id_list.insert(0, '')
prev_img_end = np.asarray([])
N_total_frames = 510
total_tp = 0
make_hyperstack = True
for i, file_id in enumerate(id_list):
    file_name = f"{file_pre}{file_id}{file_suf}"
    new_file_name = f"{file_pre}{file_id}_FIXED{file_suf}"

    print(f"Reading {file_name}")
    img = io.imread(f"{my_folder}/{file_name}")

    # flat: z, c, t

    # add the extra from the prev image to beginning
    if (len(prev_img_end) > 0):
        img = np.concatenate((prev_img_end, img))

    # Now, figure out how many extra frames are at the end
    N_frames = len(img)
    N_complete_tp = int(N_frames / (N_c * N_z))
    N_extra_frames = N_frames % (N_c * N_z)
    N_complete_frames = N_frames - N_extra_frames

    print(
        f"N_frames={N_frames}, N_complete_tp={N_complete_tp}, N_extra_frames={N_extra_frames}, N_complete_frames={N_complete_frames}")

    # Remove the extra at the end and save for next iteration
    prev_img_end = img[N_complete_frames:]
    img = img[:N_complete_frames]

    if (make_hyperstack):
        # Reshape and save the current image
        print("Reshaping")
        img = np.reshape(img, (N_z, N_c, N_complete_tp, h, w), order='F')

        t_stack = []
        for j in range(N_complete_tp):
            img_t = img[:, :, j, :, :]
            t_stack.append(img_t)
        img = None
        final_img = np.stack(t_stack, axis=0)

        print(f"Saving {new_file_name}")
        imwrite(f"{my_output_folder}/{new_file_name}",
                final_img,
                imagej=True,
                metadata={'axes': 'TZCYX'}
                )
    else:
        final_img = img
        print(f"Saving {new_file_name}")
        io.imsave(f"{my_output_folder}/{new_file_name}", final_img)

    final_img = None

    total_tp += N_complete_tp
    print(f"{total_tp} time points finished.")
