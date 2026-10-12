"""A separately authored stress test. Examples are invented, NOT collected from students.
Primary category is a subjective annotation; multi-issue requests are marked ambiguous.
Run AFTER evaluate.py so trained model exists. Does not refit any model.
"""
import csv,json
from pathlib import Path
import joblib
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
from router import keyword_predict
ROOT=Path(__file__).resolve().parent
S={
'Academic':[
('yo my advisor ghosted me how do i pick classes for next sem','slang'),
('im tryna switch from bio to comp sci who signs off','slang'),
('Can a C-minus count toward my degree audit?','unseen'),
('I missed the add/drop cutoff; who handles an exception?','unseen'),
('Where do I appeal a grade that seems wrong?','unseen'),
('Which prereqs do I need before capstone?','unseen'),
('registration says hold but I already met my adviser','ambiguous'),
('Can I enroll in two sections at the same time?','unseen'),
('How many credits left til I graduate fr','slang'),
('Need signoff for a course substitution ASAP','unseen'),
('My timetable has a class clash; can somebody fix it?','unseen'),
('Do I talk to someone about changing my program of study','unseen')],
'Technology':[
('wifi keeps kickin me off in the library','slang'),
('cant get into my account even after password reset','unseen'),
('2fa code never comes through to my phone','unseen'),
('my school mail keeps bouncing messages back','unseen'),
('bruh the campus app keeps crashing at sign in','slang'),
('The lecture site freezes whenever I submit a file','unseen'),
('is there someone who can fix a broken laptop loaner','unseen'),
('I cannot connect my tablet to eduroam','unseen'),
('printer takes my money but wont print anything','unseen'),
('the online portal just spins and never loads','unseen'),
('My transcript PDF cannot download from the student system','ambiguous'),
('I got locked out of the lab workstation','unseen')],
'Financial Aid':[
('my pell disappeared off my bill what happened','slang'),
('fafsa got processed but my award is still zero','unseen'),
('Who fixes an incorrect aid package for this term?','unseen'),
('I owe a balance after my grants were posted','unseen'),
('Why does my loan say pending disbursement','unseen'),
('can i appeal my aid bc my family income dropped','unseen'),
('Scholarship was approved but the charges remain','unseen'),
('I need emergency funding to stay enrolled','unseen'),
('When will my leftover tuition money be returned','unseen'),
('my workstudy earnings dont match what i was told','unseen'),
('my housing bill increased; can more aid cover it','ambiguous'),
('who handles missing outside scholarship checks','unseen')],
'Housing/Facilities':[
('there is water coming through my ceiling rn','slang'),
('the heater in our suite broke again','unseen'),
('my roommate moved out can i request a room swap','unseen'),
('the laundry machines on floor 4 ate my quarters','unseen'),
('There is a smell coming from the bathroom vent','unseen'),
('Can someone repair the door lock in my hall?','unseen'),
('our elevator has been out since friday','unseen'),
('there is a roach problem in the residence hall','unseen'),
('How do I report a flooded stairwell','unseen'),
('Need to move halls because of a broken window','unseen'),
('I need wifi fixed in my dorm room','ambiguous'),
('room key stopped opening my building','unseen')],
'Other':[
('where do i pick up a replacement student id','unseen'),
('who approves a new student club','unseen'),
('is there a shuttle to the stadium tomorrow','unseen'),
('how do i register a campus event','unseen'),
('where can i report a lost wallet','unseen'),
('what hours is the dining hall serving breakfast','unseen'),
('can i reserve a room for a student organization','unseen'),
('who gives permission for a bake sale outside','unseen'),
('anyone know where to get a parking permit','slang'),
('how do i join the student government','unseen'),
('Need directions to the wellness center','unseen'),
('Does the school have volunteer opportunities this week','unseen')]
}
rows=[{'message':m,'category':cat,'challenge_type':typ} for cat,rs in S.items() for m,typ in rs]
assert len(rows)==60 and len(set(r['message'] for r in rows))==60
with (ROOT/'challenge_student_messages.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=['message','category','challenge_type']);w.writeheader();w.writerows(rows)
model=joblib.load(ROOT/'model.joblib')
X=[r['message'] for r in rows];y=[r['category'] for r in rows]
# Rebuild the Naive Bayes model using the IDENTICAL training split and config in evaluate.py.
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
with (ROOT/'synthetic_student_messages.csv').open(encoding='utf-8') as f: original=list(csv.DictReader(f))
Xtr,_,ytr,_=train_test_split([r['message'] for r in original],[r['category'] for r in original],test_size=.2,stratify=[r['category'] for r in original],random_state=42)
nb=Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2))),('model',MultinomialNB())])
nb.fit(Xtr,ytr)
models={'Keyword baseline':keyword_predict(X),'Multinomial Naive Bayes':nb.predict(X).tolist(),'Logistic Regression':model.predict(X).tolist()}
results={}
for name,pred in models.items():
 results[name]={'accuracy':round(float(accuracy_score(y,pred)),4),'macro_f1':round(float(f1_score(y,pred,average='macro')),4),'n_correct':sum(a==b for a,b in zip(y,pred)),'confusion_matrix':confusion_matrix(y,pred,labels=list(S)).tolist(), 'errors':[{'message':m,'expected':t,'predicted':p} for m,t,p in zip(X,y,pred) if t!=p], 'per_class':classification_report(y,pred,output_dict=True,zero_division=0)}
with (ROOT/'challenge_results.json').open('w') as f:json.dump({'dataset':'challenge_student_messages.csv','dataset_origin':'60 newly authored hypothetical prompts with subjective primary-office labels; never used in training or tuning','n':len(y),'labels':list(S),'training_data':'synthetic_student_messages.csv (200 training/50 test split random_state=42)','results':results},f,indent=2)
with (ROOT/'challenge_predictions.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.writer(f);w.writerow(['message','expected','challenge_type','keyword','naive_bayes','logistic_regression']);w.writerows((r['message'],r['category'],r['challenge_type'],models['Keyword baseline'][i],models['Multinomial Naive Bayes'][i],models['Logistic Regression'][i]) for i,r in enumerate(rows))
for name,result in results.items():print(name,result['accuracy'],result['macro_f1'],result['n_correct'],'/60')
for e in results['Logistic Regression']['errors'][:8]:print('FAIL',e)
