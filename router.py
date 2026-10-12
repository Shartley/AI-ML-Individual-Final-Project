"""CampusHelp AI routing functions. No personal data or network required."""
import re
import joblib
from pathlib import Path

OFFICES={
 'Academic': ('Academic Advising / Registrar', 'Contact your academic advisor or the registrar for degree and registration questions.'),
 'Technology': ('Information Technology Help Desk', 'Contact campus IT support for accounts, devices, and online services.'),
 'Financial Aid': ('Office of Student Financial Aid', 'Contact financial aid for loans, grants, scholarships, and aid eligibility.'),
 'Housing/Facilities': ('Housing and Residence Life / Facilities', 'Contact housing for residential issues or facilities maintenance for repairs.'),
 'Other': ('Student Services / Campus Information', 'Ask general student services to direct you to the appropriate department.')
}
KEYWORDS={
 'Academic':['academic','advisor','degree','major','course','class','registr','transcript','graduat','prerequisite','tutoring','grade','schedule'],
 'Technology':['password','portal','email','wifi','wi-fi','authentication','laptop','computer','software','print','account locked','technical'],
 'Financial Aid':['fafsa','tuition','scholarship','loan','aid','grant','refund','work study','balance','funding'],
 'Housing/Facilities':['dorm','roommate','residence','housing','maintenance','leak','sink','window','elevator','mold','hallway','air conditioner','room assignment'],
 'Other':['club','event','volunteer','parking','id','dining','lost','activities','tour','recreation','campus']
}

def keyword_predict(messages):
 results=[]
 for message in messages:
  m=message.lower()
  scores={k:sum(bool(re.search(r'\b'+re.escape(t)+r'\b',m)) for t in terms) for k,terms in KEYWORDS.items()}
  results.append(max(scores,key=scores.get) if max(scores.values()) else 'Other')
 return results

def route(message, model=None, threshold=.60):
 if not message.strip(): return {'error':'Please enter a question.'}
 if model is None:
  model=joblib.load(Path(__file__).parent/'model.joblib')
 probabilities=model.predict_proba([message])[0]
 ranked=sorted(zip(model.classes_,probabilities),key=lambda p:-p[1])
 label,conf=ranked[0]
 office,description=OFFICES[label]
 uncertain=float(conf)<threshold
 return {'category':label,'confidence':round(float(conf),3),'office':office,
         'guidance':description,'review_needed':uncertain,
         'status':'Human review recommended — low confidence' if uncertain else 'Suggested routing (verify with your university)',
         'alternatives':[(x,round(float(p),3)) for x,p in ranked[1:3]]}
