from pathlib import Path
import csv, random
root=Path(__file__).parent
categories={
'Academic': {'subjects':['my academic advisor','degree requirements','changing my major','course registration','a prerequisite override','my transcript','graduation requirements','tutoring services','an incomplete grade','my class schedule'], 'phrases':['Who can help me with {}?','I have a question about {}.','Where should I go to discuss {}?','Could someone explain {}?','I need assistance with {}.']},
'Technology': {'subjects':['my campus password','the student portal','my university email','connecting to campus Wi-Fi','two factor authentication','a broken laptop','the learning management system','printing from my computer','software installation','my account being locked'], 'phrases':['Who fixes issues with {}?','I cannot access {}.','I need technical support for {}.','Where can I get help with {}?','Something is wrong with {}.']},
'Financial Aid': {'subjects':['my FAFSA','my tuition bill','my scholarship payment','a student loan','financial aid verification','an unexpected account balance','work study funding','my grant award','my refund','my aid package'], 'phrases':['Who can explain {}?','I need help understanding {}.','Where do I ask about {}?','There is a problem with {}.','Can someone review {}?']},
'Housing/Facilities': {'subjects':['my dorm air conditioner','the broken window in my residence hall','a leaking dorm sink','a roommate change','campus maintenance','my room assignment','the elevator in my dorm','a residence hall key','mold in my room','a broken hallway light'], 'phrases':['Who can fix {}?','I need to report {}.','Where do I request help with {}?','There is a problem with {}.','Can someone look into {}?']},
'Other': {'subjects':['joining a campus club','student events','volunteer opportunities','a campus parking permit','getting a student ID','campus dining options','a lost item','student activities','a campus tour','the recreation center'], 'phrases':['Where can I learn about {}?','Who handles questions about {}?','Can you tell me about {}?','I need information on {}.','Which office handles {}?']}
}
rows=[]
for label, d in categories.items():
 for phrase in d['phrases']:
  for subject in d['subjects']:
   rows.append({'message':phrase.format(subject), 'category':label})
random.Random(42).shuffle(rows)
with open(root/'synthetic_student_messages.csv','w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=['message','category']);w.writeheader();w.writerows(rows)
print('Generated',len(rows),'synthetic messages')
