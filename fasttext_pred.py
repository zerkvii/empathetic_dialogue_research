import fasttext
import numpy as np
from tqdm import tqdm

ftmodel = fasttext.FastText.load_model('./save/fasttext_empathetic_dialogues.mdl')

d = {}
# train dev test
option='test'
d['context'] = np.load(f'empdial_dataset/sys_dialog_texts.{option}.npy', allow_pickle=True)
d['target'] = np.load(f'empdial_dataset/sys_target_texts.{option}.npy', allow_pickle=True)
d['emotion'] = np.load(f'empdial_dataset/sys_emotion_texts.{option}.npy', allow_pickle=True)

preds = []
for i in tqdm(range(len(d['emotion']))):
    t = " </s> ".join(d['context'][i])
    pred, _ = ftmodel.predict(t, k=1)
    pred_emo = pred[0].split('__')[-1]
    preds.append(pred_emo)
    # print(t)
    # print(d['emotion'][i])
    # print(pred_emo)
    # print('---------')

np.save(f'./empdial_dataset/fasttest_pred_emotion_texts.{option}.npy', preds, allow_pickle=True)