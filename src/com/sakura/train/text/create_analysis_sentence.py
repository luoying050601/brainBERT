import os
import pandas as pd
from nltk.tokenize import sent_tokenize

DATA_DIR = os.path.abspath(os.path.join(os.getcwd(), "../../../../../"))

# Import the Dataset and preprocess
alice = open(DATA_DIR + "/original_data/text/Alice's wonderland.txt", 'rb')


# def sentence_split(str_centence):
#     list_ret = list()
#     for s_str in str_centence.split('.'):
#         if '?' in s_str:
#             list_ret.extend(s_str.split('?'))
#         elif '!' in s_str:
#             list_ret.extend(s_str.split('!'))
#         else:
#             list_ret.append(s_str)
#     return list_ret

# better performance
def sentence_token_nltk(str):
    sent_tokenize_list = sent_tokenize(str)
    return sent_tokenize_list


lines = []
for line in alice:
    line = line.strip().lower()
    line = line.decode('ascii', 'ignore')
    if len(line) == 0:
        continue
    lines.append(line)
alice.close()
text = " ".join(lines)
# p = splitParagraphIntoSentences(text)
# lines = sentence_split(text)  # a list of all sentences
lines_nltk = sentence_token_nltk(text)  # a list of all sentences
# print(lines_nltk)
# df = pd.DataFrame({'sentence': lines})
df = pd.DataFrame({'sentence': lines_nltk})
# print(df)
# df.to_csv(DATA_DIR + '/output/BERT/Alice_data.tsv')
df.to_csv(DATA_DIR + '/output/BERT/Alice_data.tsv', index=False)
