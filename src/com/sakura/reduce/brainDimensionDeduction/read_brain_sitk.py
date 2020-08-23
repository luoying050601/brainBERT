#  in use
import numpy as np
import SimpleITK as sitk
import os
import pickle


def Standardization(X, mean, sd):
    return float(mean - X) / sd


DATA_DIR = os.path.abspath(os.path.join(os.getcwd(), "../../../../../"))
# subjects = ['func', 'derivatives', 'anat']
subjects = ['derivatives']
# participants = ['18', '22', '23', '24', '26', '28', '30', '31', '35', '36', '37', '39',
#                 '41', '42', '43', '44', '45', '47', '48', '49', '50', '51', '53']

participants = ['48']
file_name = {
    'derivatives': 'task-alice_bold_preprocessed',
    'func': 'task-alice_bold',
    'anat': 'T1w'
}
for user in participants:
    for sub in subjects:
        print(user, sub)
        # img = sitk.ReadImage(DATA_DIR + '/original_data/fMRI/sub-18_task-alice_bold_preprocessed.nii.gz')
        img = sitk.ReadImage(DATA_DIR + '/original_data/fMRI/sub-' + user + '/'
                             + sub + '/sub-' + user + '_' + file_name[sub] + '.nii.gz')
        img = sitk.GetArrayFromImage(img)
        # (372, 68, 95, 79)
        #print(img.shape)
        mean = np.mean(img)
        sd = np.std(img)
        # # Standardization
        f2 = np.vectorize(Standardization)
        new_data = np.array([[f2(X, mean, sd) for X in row] for row in img])
        # normalization
        min_data = np.min(new_data)
        max_data = np.max(new_data)
        new_data = (new_data - min_data) / (max_data - min_data)

        new_arr = new_data.reshape((372, 68*95*79), order='F')

           # save
        with open(DATA_DIR + '/output/fMRI/sub-'+user+'/sub-'+user+'_'+file_name[sub]+'.pickle', 'wb') as f:
            pickle.dump(new_arr, f)


