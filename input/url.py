from  django.urls import path,include
from . import views

urlpatterns = [
    path('',views.input,name="input"),
    path('result/',include('result.url'))
]