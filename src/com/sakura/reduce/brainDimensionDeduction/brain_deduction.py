from nilearn.input_data import NiftiMasker
import os

DATA_DIR = os.path.abspath(os.path.join(os.getcwd(), "../../../../../"))
# what for?
threshold = 0.70
subjects = ['derivatives']
#
participants = ['18', '22', '23', '24', '26', '28', '30', '31', '35', '36', '37', '39',
                '41', '42', '43', '44', '45', '47', '48', '49', '50', '51', '53']
# participants = ['18']
file_name = {
    'derivatives': 'task-alice_bold_preprocessed',
    'func': 'task-alice_bold',
    'anat': 'T1w'
}
for user in participants:
    for sub in subjects:
        nifti_filename = DATA_DIR + '/original_data/fMRI/sub-' + user + '/'+sub+'/sub-' + user + '_' + file_name[sub] + '.nii.gz'
        masker = NiftiMasker()
        mask = masker.fit(nifti_filename)
        masked_data = masker.fit_transform(nifti_filename)
        print(masked_data.shape)
