from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('test/', views.url_test, name='url_test'),
    path('new_entries/', views.new_entries, name='new_entries'),
    path('hello/', views.url_check, name='hello'),
            ]