from pathlib import Path
import gradio as gr
import joblib
from router import route

MODEL=joblib.load(Path(__file__).parent/'model.joblib')
def classify(question, threshold):
 answer=route(question,MODEL,threshold)
 if 'error' in answer:return answer['error'],'','',''
 return (f"### {answer['office']}\n{answer['guidance']}",
         f"**Category:** {answer['category']}  |  **Model probability:** {answer['confidence']:.1%}",
         f"**{answer['status']}**",
         '\n'.join(f'{c}: {p:.1%}' for c,p in answer['alternatives']))
with gr.Blocks(title='CampusHelp AI') as demo:
 gr.Markdown('# 🎓 CampusHelp AI\n### Find the right campus support office\nThis research prototype uses synthetic training examples. Predictions are suggestions, not official university directions. Do not enter personal information.')
 question=gr.Textbox(label='What do you need help with?',placeholder='Example: Who can help me with my scholarship not showing up?',lines=3)
 threshold=gr.Slider(.2,.95,value=.6,step=.05,label='Low-confidence human-review threshold')
 button=gr.Button('Find My Office',variant='primary')
 office=gr.Markdown();category=gr.Markdown();status=gr.Markdown();alternatives=gr.Textbox(label='Other possible categories')
 button.click(classify,[question,threshold],[office,category,status,alternatives]);question.submit(classify,[question,threshold],[office,category,status,alternatives])
 gr.Examples(['I cannot log into my campus account','My scholarship disappeared from my tuition bill','The window in my dorm is broken','Who can explain the requirements for my degree?','My laptop is broken and I need to pay for a replacement'],inputs=question)
 gr.Markdown('**Privacy:** Please do not enter names, student IDs, financial account numbers, or health information.')
if __name__=='__main__':demo.launch()
