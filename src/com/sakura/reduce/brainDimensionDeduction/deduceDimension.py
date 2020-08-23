import pickle
import numpy as np
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


def reduce_dimention(brain_data, ac_data, threshold):
    # ROIの相関が高い領域だけを抽出.(ある一定の値以上のものを調べる)
    Brain_reduced = []
    for i, cor in enumerate(ac_data):
        # print(cor)
        if cor > threshold:
            Brain_reduced.append(brain_data[:, i])
    Brain_reduced = np.array(Brain_reduced)
    Brain_reduced = Brain_reduced.T

    return Brain_reduced


def main():
    for user in participants:
        for sub in subjects:
            with open(DATA_DIR + '/output/fMRI/sub-' + user + '/sub-' + user + '_' + file_name[sub] + '_acc.pickle',
                      'rb') as f1:
                ac_data = pickle.load(f1, encoding='latin1')

            with open(DATA_DIR + '/output/fMRI/sub-' + user + '/sub-' + user + '_' + file_name[sub] + '.pickle',
                      'rb') as f2:
                test_data = pickle.load(f2, encoding='latin1')
            print('削減前')
            print(test_data.shape)

            test_reduced = reduce_dimention(test_data, ac_data, threshold)

            print('削減後')
            print(test_reduced.shape)

            with open(DATA_DIR + '/output/fMRI/sub-' + user + '/sub-' + user + '_' + file_name[sub] + '_reduced_' + str(
                    threshold) + '.pickle',
                      'wb') as f:
                pickle.dump(test_reduced, f)


if __name__ == '__main__':
    main()
