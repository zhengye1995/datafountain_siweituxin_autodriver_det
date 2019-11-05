txt1 = 'data/dataset1/train.txt'
txt2 = 'data/DF_1018/train.txt'
save_txt = 'data/siweituxin/Annotations/train.txt'
with open(txt1, 'r') as f1:
    label1 = f1.readlines()

with open(txt2, 'r') as f2:
    label2 = f2.readlines()

label1.extend(label2)

with open(save_txt, 'w') as f:
    f.writelines(label1)