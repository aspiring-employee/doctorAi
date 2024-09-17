from django.shortcuts import render
from django.http import HttpResponse
from .forms import fileupload
import os
import PyPDF2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
import pytesseract
from pdf2image import convert_from_path
from dotenv import load_dotenv

load_dotenv()


# Create your views here.
def find_value(i,text):
    
    index=text.find(i)
    index=index+len(i)
    end=index
    while text[end] != '\n':
        end+=1
    li_value = text[index:end]
    value_list = li_value.split() 
    if(value_list[0]==':'):
        value_list.remove(value_list[0])
    temp_value = value_list[0]
    temp_dict={"m":1,"f":0,"asymptomatic":0,"atypical":1,
               "non-anginal":2,"typical":3,"yes":1,"no":0,
               "down":0,"down-sloping":0,"flat":1,"upsloping":2,"fixed":0,"fixed-defect":1,"normal":2,"reversible":3}
    if(temp_value[-1]==';'):
        temp_value=temp_value[:-1]
    new_temp_value=temp_value
    if(temp_value in list(temp_dict.keys())):
        new_temp_value=temp_dict[temp_value]
    return new_temp_value


def handle_uploaded_file(f):  
    with open('media/'+f.name, 'wb+') as destination:  
        for chunk in f.chunks():  
            destination.write(chunk)


def input(request):
    if(request.method == 'POST'):
        form = fileupload(request.POST,request.FILES)
        file = request.FILES['file']
        file_name= 'media/' + file.name
        handle_uploaded_file(file)
        #-----------------
        df=pd.read_csv("Dataset/Heart_minor.csv",index_col=0)

        # pdf = open(file_name,"rb")
        # reader = PyPDF2.PdfReader(pdf)
        # text=""
        # page=reader.pages[0]
        # text=page.extract_text().lower()

        # # Path to the PDF file
        # file_name = 'your_pdf_file.pdf'

        # # Convert PDF to images
        pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
        pages = convert_from_path(file_name, 500)#, poppler_path = r"poppler-24.07.0/Library/bin")
        text = ""
        if pages:
            text = pytesseract.image_to_string(pages[0]).lower() 
        print(text)
        
        attributes = ["age","sex","chest_pain","cholesterol","fasting","thalach","angina","old_peak","slope","ca","thalassemia","target"]
        df.columns = attributes
        attributes = attributes[:-1]

        from pydantic import BaseModel, Field
        from langchain.output_parsers import PydanticOutputParser
        from langchain_core.prompts.chat import ChatPromptTemplate
        from langchain_mistralai.chat_models import ChatMistralAI

        class Features(BaseModel):
            age: int = Field(description="age of the patient; if not present then -1")
            sex: int = Field(description="male:1, female:0, none:-1")
            chest_pain: int = Field(description="asymptomatic:0, atypical:1, non-anginal:2, typical:3, none:-1")
            cholesterol: float = Field(description="cholestrol level of the patient; if not present then -1")
            fasting: int = Field(description="yes:1, no:0, none:-1")
            thalach: float = Field(description="thalach measure of the patient; if not present then -1")
            angina: int = Field(description="yes:1, no:0, none:-1")
            old_peak: float = Field(description="old peak value of the patient; if not present then -1")
            slope: int = Field(description="down:0, down-sloping:0, flat:1, upsloping:2, none:-1")
            ca: int = Field(description="major vessels for the patient; if not present then -1")
            thalassemia: int = Field(description="fixed:0, fixed-defect:1, normal:2, reversible:3")

            def asdict(self):
                return {
                    'age': self.age,
                    'sex': self.sex,
                    'chest_pain': self.chest_pain,
                    'cholesterol': self.cholesterol,
                    'fasting': self.fasting,
                    'thalach': self.thalach,
                    'angina': self.angina,
                    'old_peak': self.old_peak,
                    'slope': self.slope,
                    'ca': self.ca,
                    'thalassemia': self.thalassemia,
                }
        
        parser = PydanticOutputParser(pydantic_object=Features)
        format_instructions = parser.get_format_instructions()
        template = """You are an experienced healthcare professional. You will be given a heart report. \
            Take the below report delimited by triple backticks and extract the relevant information.
            Report: ```{text}```
            
            Format Instructions: {format_instructions}"""
        prompt = ChatPromptTemplate.from_template(template=template)
        message = prompt.format_messages(text=text, format_instructions=format_instructions)

        llm = ChatMistralAI(
            temperature=0,
        )
        output = llm.invoke(message)
        values = parser.parse(output.content).asdict()

        # values = {}
        # for i in attributes:
        #     if(i in text):
        #         value = find_value(i,text)
        #         values[i]=float(value)
        #     else:
        #         values[i]=-1

        for i in values.keys():
            if(values[i]==-1):
                df.drop(columns=[i],inplace=True)

        x = df.iloc[:,:-1].values
        y = df.iloc[:,-1].values
        rand_int = np.random.randint(1,100)
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
        ss = StandardScaler()
        x_train_ss = ss.fit_transform(x_train)
        x_test_ss = ss.transform(x_test)

        #catboost
        model = CatBoostClassifier(depth=6,iterations=30,learning_rate=0.1)
        model.fit(x_test_ss,y_train)
        y_pred=model.predict(x_test_ss)
        input_list = list(map(float,[values[x] for x in values.keys() if(values[x])!=-1]))
        final_label = model.predict(ss.transform([input_list]))
        final_acc = accuracy_score(y_test,y_pred)
        
        if(final_label==1):
            output="HAVE"
        else:
            output = "DON'T HAVE"
        output=f"\nWith {final_acc}% Accuracy, our Doctor Believes that You {output} a Heart Disease."
        data={"temp":output}
        return render(request,'result/result.html',data)
    else:
        form = fileupload()
        return render(request, 'input/input.html')




    
    



    

