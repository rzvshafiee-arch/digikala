import pandas as pd
from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns
import  matplotlib.pyplot  as plt 
from sklearn.metrics import accuracy_score
from sklearn.metrics import  precision_score




data=pd.read_excel("test.xlsx")

predictions=data["prediction"].values
truth=data["Suggestion"].values

cm=confusion_matrix(truth,predictions,labels=[1,2,3])

plt.figure(figsize=(5,5))
sns.heatmap(cm,annot=True,xticklabels=[1,2,3],yticklabels=[1,2,3],fmt="d")


print( accuracy_score(truth,predictions))

print(precision_score(truth, predictions, average=None,
                      labels=[1,2,3]))

print(classification_report(truth,predictions))

plt.show()