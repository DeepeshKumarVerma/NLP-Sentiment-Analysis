from django.urls import path
from . import views # import your view 

urlpatterns= [
    path('', views.sentiment_view, name= 'home'),  # default route
    path('predict/', views.sentiment_view, name= 'predict'), # # or use separate prediction route
]