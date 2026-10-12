"""Train/evaluate 3 baseline models using one fixed held-out split."""
import csv,json,platform
from pathlib import Path
import numpy as np
import joblib
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
from router import keyword_predict
from build_project import root

with open(root/'synthetic_student_messages.csv',encoding='utf-8') as f: rows=list(csv.DictReader(f))
X=[r['message'] for r in rows];y=[r['category'] for r in rows]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
models={'Keyword baseline':None,
        'Multinomial Naive Bayes':Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2))),('model',MultinomialNB())]),
        'Logistic Regression':Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2))),('model',LogisticRegression(max_iter=2000,random_state=42))])}
results={}; predictions={}
for name,model in models.items():
 if model is None: pred=keyword_predict(Xte)
 else: model.fit(Xtr,ytr);pred=model.predict(Xte).tolist()
 predictions[name]=pred
 results[name]={'accuracy':round(float(accuracy_score(yte,pred)),4),'macro_f1':round(float(f1_score(yte,pred,average='macro')),4),
                'per_class':classification_report(yte,pred,output_dict=True,zero_division=0),
                'confusion_matrix':confusion_matrix(yte,pred,labels=sorted(set(y))).tolist(),
                'errors':[{'message':m,'true':t,'predicted':p} for m,t,p in zip(Xte,yte,pred) if t!=p]}
 # records each model fitted only to training data
 if name=='Logistic Regression':joblib.dump(model,root/'model.joblib')
output={'dataset':'synthetic_student_messages.csv','data_origin':'Newly generated synthetic prompts from deterministic templates; not the prior checkpoint dataset.',
        'n_rows':len(rows),'train_rows':len(Xtr),'test_rows':len(Xte),'random_state':42,
        'hardware':platform.platform()+'; '+platform.processor(), 'python':platform.python_version(),'sklearn':sklearn.__version__,
        'model': 'Logistic Regression for demo','results':results}
(root/'results.json').write_text(json.dumps(output,indent=2))
with open(root/'heldout_predictions.csv','w',newline='',encoding='utf-8') as f:
 w=csv.writer(f);w.writerow(['message','true','keyword_pred','naive_bayes_pred','logistic_regression_pred']);w.writerows(zip(Xte,yte,predictions['Keyword baseline'],predictions['Multinomial Naive Bayes'],predictions['Logistic Regression']))
print('250-message reproducibility experiment (NOT original checkpoint):')
for name,v in results.items(): print(name, v['accuracy'],v['macro_f1'],'errors:',len(v['errors']))
