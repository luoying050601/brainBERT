import json
import os

DATA_DIR = os.path.abspath(os.path.join(os.getcwd(), "../../../../../"))

json_filename = DATA_DIR + '/original_data/json/sub-30/sub-30_task-alice_bold.json'  # 这是json文件存放的位置
# txt_filename = '/home/hjxu/AI_Challenger-master/code_xu/12.02/finnal.txt'  # 这是保存txt文件的位置
# file = open(txt_filename, 'w')
with open(json_filename) as f:
    pop_data = json.load(f)
    SliceTiming = pop_data['SliceTiming']
    print(SliceTiming)
    print(len(SliceTiming))

    # image_id = pop_dict['image_id']
    # file.write(temp + '\n')
# file.close()
# with open(submit, 'w') as f:
#     json.dump(result, f)