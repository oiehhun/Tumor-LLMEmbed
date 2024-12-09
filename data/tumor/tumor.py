import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# Data Load
data = pd.read_excel('../WholeSpine700.xlsx')

# Data Preprocessing
for i, label in enumerate(data['GT_label']):
    if label == 'no ' or label == 'No':
        data['GT_label'][i] = 'no'
        
# Train Test Split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
print(f'훈련 데이터셋 수 : {len(train_data)}')
print(f'테스트 데이터셋 수 : {len(test_data)}')


# Label Encoding
num_labels = {'no':0, 'mets':1, 'progression':2, 'stable':3, 'improved':4, 'romets':5}
train_data["label"] = 0
test_data["label"] = 0

for keyStr in num_labels.keys():
    train_data["label"][train_data["GT_label"]==keyStr] = num_labels[keyStr]
    test_data["label"][test_data["GT_label"]==keyStr] = num_labels[keyStr]

train_df = train_data[['Reports', 'label']]
test_df = test_data[['Reports', 'label']]

train_df.to_csv('./tumor/tumor_train.csv', index=False)
test_df.to_csv('./tumor/tumor_test.csv', index=False)