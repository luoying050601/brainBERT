# in use
import os
import pickle
from nilearn.image import mean_img
import numpy as np

DATA_DIR = os.path.abspath(os.path.join(os.getcwd(), "../../../../../"))

# file_path3D = DATA_DIR + '/original_data/fMRI/sub-18/anat/sub-18_T1w.nii.gz'
# total 23
participants = ['18', '22', '23', '24', '26', '28', '30', '31', '35', '36', '37', '39',
                '41', '42', '43', '44', '45', '47', '48', '49', '50', '51', '53']
# participants = ['48']

file_name = {
    'derivatives': 'task-alice_bold_preprocessed',
    'func': 'task-alice_bold',
    'anat': 'T1w'
}
subjects = ['derivatives']

for sub in subjects:
    for user in participants:
        file_path = DATA_DIR + '/original_data/fMRI/sub-' + user + '/derivatives/sub-' + user + '_' + file_name[
            sub] + '.nii.gz'

        meanImg = mean_img(file_path).get_fdata()
        _range = np.max(meanImg) - np.min(meanImg)
        new_data = (meanImg - np.min(meanImg)) / _range
        new_arr = new_data.reshape((68 * 95 * 79, 1), order='F')
        # print(new_arr, new_arr.shape)
        with open(DATA_DIR + '/output/fMRI/sub-' + user + '/sub-' + user + '_' + file_name[sub] + '_acc.pickle',
                  'wb') as f:
            pickle.dump(new_arr, f)
