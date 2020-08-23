import matplotlib.pyplot as plt
import os
import nibabel as nib
import pickle

# img = nib.load('../../../original_data/fMRI/sub-18_T1w.nii.gz')
# (176, 265, 265)

DATA_DIR = os.path.abspath(os.path.join(os.getcwd(), "../../../../../"))

nii = nib.load(DATA_DIR + '/original_data/fMRI/sub-18_task-alice_bold_preprocessed.nii.gz')
img_np = nii.get_fdata()

print('#########################')
# # (79, 95, 68, 372)
# print(img_np.shape)
print(img_np)

new_arr = img_np.reshape((79 * 95 * 68, 372), order='F')
# print('#########################')
# print(new_arr.shape)
# print(new_arr)
#
# with open(DATA_DIR + '/output/sub-18_task-alice_bold_preprocessed.pickle', 'wb') as f:
#     pickle.dump(new_arr, f)

# 515712 (79, 95, 68, 372)
# (72, 72, 44, 374) 228096

# img_affine = nii.affine
#
# new_image = nib.Nifti1Image(img_np, img_affine)
# print(new_image)
# print(new_image.shape)
# print(img_affine.shape)

# nib.save(new_image, '../../../original_data/fMRI/sub-18_task-alice_bold_preprocessed_later.nii.gz')
