from django import forms 

class fileupload(forms.Form):
    file = forms.FileField()