from django.shortcuts import render
from django.http import HttpResponse
from .forms import fileupload
import os
import PyPDF2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
import pytesseract
from pdf2image import convert_from_path


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
               "down":0,"down-sloping":0,"flat":1,"upsloping":2,"fixed":1,"fixed-defect":1,"normal":2,"reversible":3}
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

        pdf = open(file_name,"rb")
        reader = PyPDF2.PdfReader(pdf)
        text=""
        page=reader.pages[0]
        text=page.extract_text().lower()

        # # Path to the PDF file
        # file_name = 'your_pdf_file.pdf'

        # # Convert PDF to images
        pytesseract.pytesseract.tesseract_cmd = 'tess/tesseract.exe'
        pages = convert_from_path(file_name, 500)#, poppler_path = r"poppler-24.07.0/Library/bin")
        text = ""
        if pages:
            text = pytesseract.image_to_string(pages[0]).lower()
        print(text)
        
        attributes = ["age","sex","chest pain","cholesterol","(fasting)","thalach","angina:","old-peak","slope","(ca)","thalassemia","target"]
        df.columns = attributes
        attributes = attributes[:-1]

        values = {}
        for i in attributes:
            if(i in text):
                value = find_value(i,text)
                values[i]=float(value)
            else:
                values[i]=-1

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
        
        # knn classifier
        KNN = KNeighborsClassifier(n_neighbors=13,metric="minkowski",p=2)
        KNN.fit(x_train_ss,y_train)
        y_pred = KNN.predict(x_test_ss)

        input_list = list(map(float,[values[x] for x in values.keys() if(values[x])!=-1]))

        label = KNN.predict(ss.transform([input_list]))
        print(label)

        # random forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(x_train_ss, y_train)
        y_pred = rf.predict(x_test_ss)

        input_list = list(map(float, [values[x] for x in values.keys() if values[x] != -1]))
        label = rf.predict(ss.transform([input_list]))

        # gb classifier
        gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42)
        gbc.fit(x_train_ss, y_train)
        y_pred = gbc.predict(x_test_ss)

        input_list = list(map(float, [values[x] for x in values.keys() if values[x] != -1]))
        label = gbc.predict(ss.transform([input_list]))

        # xgb
        xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        xgb_model.fit(x_train_ss, y_train)
        y_pred = xgb_model.predict(x_test_ss)
        input_list = list(map(float, [values[x] for x in values.keys() if values[x] != -1]))
        label = xgb_model.predict(ss.transform([input_list]))
        
        if(label==0):
            output="DON'T HAVE"
        if(label==1):
            output = "HAVE"
        output=f"\nWith {round(accuracy_score(y_test,y_pred)*100,2)}% Accuracy, our Doctor Believes that You {output} a Heart Disease."
        data={"temp":output}
        return render(request,'result/result.html',data)
    else:
        form = fileupload()
        return render(request, 'input/input.html')




    
    



    

