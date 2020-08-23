import pickle
import os
subjects = ['derivatives']
participants = ['18']
file_name = {
    'derivatives': 'task-alice_bold_preprocessed',
    'func': 'task-alice_bold',
    'anat': 'T1w'
}

DATA_DIR = os.path.abspath(os.path.join(os.getcwd(), "../../../../../"))
for user in participants:
    for sub in subjects:
        print(user, sub)
        with open(DATA_DIR + '/output/fMRI/sub-'+user+'/sub-'+user+'_'+file_name[sub]+'.pickle', 'rb') as f:
            test_data = pickle.load(f, encoding='latin1')
        print(test_data)
        print('###########################')
