import pandas as pd
import numpy as np
from sklearn import metrics, cluster, mixture
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier
from itertools import cycle
from proj import plot, evaluation as eval, datapreprocessing as datapp
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 15)
rs = 32
data = pd.read_csv('data/pd_speech_features.csv', skiprows=[0])
to_clf = "class"
categoric = ["gender", "id"]
to_remove = ["id"]
data.shape

normalization = "minmax"
bal = "smote"
df = datapp.preprocess(data, to_clf, normalization=normalization, ignore_classes=categoric, as_df=True)
df=data
y: np.ndarray = df[to_clf].values
X: np.ndarray = df.drop(to_clf, axis=1).values
#%%
select = SelectKBest(f_classif, k=10).fit(X, y)
ind = select.get_support(indices=True)
col = df.columns[ind].tolist()

X_new = select.transform(X)
dfk = pd.DataFrame(X_new, columns=col)
#%%
bins = list(range(3,12))
qdfs = []
cdfs = []
for b in bins:
    qdfs.append(eval.cut(dfk, b, ['class','id', 'gender'], cut="qcut"))
    cdfs.append(eval.cut(dfk, b, ['class','id', 'gender'], cut="cut"))
#%%
dummy_qdfs = []
dummy_cdfs = []
for i in range(len(bins)):
    dummy_qdfs.append(eval.dummy(qdfs[i], ['class','id','gender']))
    dummy_cdfs.append(eval.dummy(cdfs[i], ['class', 'id', 'gender']))
#%%
fiq_q = []
fiq_c =[]
for i in range(len(bins)):
    fiq_q.append(eval.freq_itemsets(dummy_qdfs[i], minpaterns=100))
    fiq_c.append(eval.freq_itemsets(dummy_cdfs[i], minpaterns=100))
#%%
rules_q = []
rules_c = []
for i in range(len(bins)):
    rules_q.append(eval.assoc_rules(fiq_q[i], orderby="lift", min_confidence=0.9).head(20))
    rules_c.append(eval.assoc_rules(fiq_c[i], orderby="lift", min_confidence=0.9).head(20))

rules_qsup = []
rules_csup = []
for i in range(len(bins)):
    rules_qsup.append(eval.assoc_rules(fiq_q[i], orderby="support",inverse=False, min_confidence=0.9).head(20))
    rules_csup.append(eval.assoc_rules(fiq_c[i], orderby="support", inverse=False,min_confidence=0.9).head(20))

#%%
q_lifts = []
c_lifts = []
for i in range(len(bins)):
    q_lifts.append(rules_q[i]["lift"].mean())
    c_lifts.append(rules_c[i]["lift"].mean())

q_sup = []
c_sup = []
for i in range(len(bins)):
    q_sup.append(rules_qsup[i]["support"].mean())
    c_sup.append(rules_csup[i]["support"].mean())
#%%
lvalues={}
lvalues["cut"] = c_lifts
lvalues["qcut"] = q_lifts

svalues = {}
svalues["cut"] = c_sup
svalues["qcut"] = q_sup

plt.figure()
fig, axs = plt.subplots(1, 2, figsize=(12, 6), squeeze=False)
axs[0,0].set_xticks(bins)
axs[0,1].set_xticks(bins)
plot.multiple_line_chart(axs[0, 0], bins, lvalues, 'Lift of top rules of corresponding bins','bins', 'lift')
plot.multiple_line_chart(axs[0, 1], bins, svalues, 'Support of top rules of corresponding bins','bins', 'support')

plt.show()




"""

#%%
df_qcut7 = eval.cut(dfk, 7, ['class','id', 'gender'], cut="qcut")
df_qcut5 = eval.cut(dfk, 5, ['class','id', 'gender'], cut="qcut")
df_qcut3 = eval.cut(dfk, 3, ['class','id', 'gender'], cut="qcut")
df_cut7 = eval.cut(dfk, 7, ['class','id', 'gender'], cut="cut")
df_cut5 = eval.cut(dfk, 5, ['class','id', 'gender'], cut="cut")
df_cut3 = eval.cut(dfk, 3, ['class','id', 'gender'], cut="cut")

#%%
dummqc7 = eval.dummy(df_qcut7, ['class','id','gender'])
dummqc5 = eval.dummy(df_qcut5, ['class','id','gender'])
dummqc3 = eval.dummy(df_qcut3, ['class','id','gender'])
dummc7 = eval.dummy(df_cut7, ['class','id','gender'])
dummc5 = eval.dummy(df_cut5, ['class','id','gender'])
dummc3 = eval.dummy(df_cut3, ['class','id','gender'])

#%%
fiqc7 = eval.freq_itemsets(dummqc7, minpaterns=100)
fiqc5 = eval.freq_itemsets(dummqc5, minpaterns=100)
fiqc3 = eval.freq_itemsets(dummqc3, minpaterns=100)
print("\n-------------------------------------------")
fic7 = eval.freq_itemsets(dummc7, minpaterns=100)
fic5 = eval.freq_itemsets(dummc5, minpaterns=100)
fic3 = eval.freq_itemsets(dummc3, minpaterns=100)

#%%
rulesqc7 = eval.assoc_rules(fiqc7, orderby="lift", min_confidence=0.9)
rulesqc5 = eval.assoc_rules(fiqc5, orderby="lift", min_confidence=0.9)
rulesqc3 = eval.assoc_rules(fiqc3, orderby="lift", min_confidence=0.9)
rulesc7 = eval.assoc_rules(fic7, orderby="lift", min_confidence=0.9)
rulesc5 = eval.assoc_rules(fic5, orderby="lift", min_confidence=0.9)
rulesc3 = eval.assoc_rules(fic3, orderby="lift", min_confidence=0.9)

#%%
rulesqc7top = eval.assoc_rules(fiqc7, orderby="lift", min_confidence=0.9).head(20)
rulesqc5top = eval.assoc_rules(fiqc5, orderby="lift", min_confidence=0.9).head(20)
rulesqc3top = eval.assoc_rules(fiqc3, orderby="lift", min_confidence=0.9).head(20)
rulesc7top = eval.assoc_rules(fic7, orderby="lift", min_confidence=0.9).head(20)
rulesc5top = eval.assoc_rules(fic5, orderby="lift", min_confidence=0.9).head(20)
rulesc3top = eval.assoc_rules(fic3, orderby="lift", min_confidence=0.9).head(20)

#%%
qc7top = rulesqc7top["lift"].mean()
qc5top =rulesqc5top["lift"].mean()
qc3top = rulesqc3top["lift"].mean()
c7top = rulesc7top["lift"].mean()
c5top = rulesc5top["lift"].mean()
c3top = rulesc3["lift"].mean()
#%%
values={}
values["cut"] = [c3top, c5top, c7top]
values["qcut"] = [qc3top, qc5top, qc7top]
bins =[3,5,7]

plt.figure()
fig, axs = plt.subplots(1, 1, figsize=(10, 8), squeeze=False)
axs[0,0].set_xticks(bins)
plot.multiple_line_chart(axs[0, 0], bins, values, 'Decision Trees with %s min_samples_split' ,
                         'bins', 'lift')

plt.show()
"""