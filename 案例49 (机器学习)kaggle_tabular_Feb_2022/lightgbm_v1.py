import lightgbm as lgb
import pandas as pd
import pickle


def self_callbacks(cb):
    print(cb)

print("LGB test")
clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=55, reg_alpha=0.0, reg_lambda=1,
        max_depth=15, n_estimators=6000, objective='binary',
        subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
        learning_rate=0.06, min_child_weight=1, random_state=20, n_jobs=-1
    )

X = pd.read_csv('data/train_data.csv')
label = pd.read_csv('data/train_label.csv')
y = label.target

clf.fit(X, y, callbacks=[lgb.log_evaluation(period=1, show_stdv=True)])
#pre=clf.predict(testdata)

# 保存模型
s=pickle.dumps(clf)
f=open('lightgbm_v2.model', "wb+")
f.write(s)
f.close()