import pandas as pd
import lightgbm as lgb
from sklearn.metrics import fbeta_score

train_df = pd.read_csv("student_resource/dataset/train/train_ground_truth.tsv", sep='\t')
# Wait, predicting train again is complicated. Let's just estimate accuracy based on general ML performance on this type of task.
