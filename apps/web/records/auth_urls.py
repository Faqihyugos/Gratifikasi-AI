"""URL configuration for auth endpoints."""
from django.urls import path
from . import auth_views

urlpatterns = [
    path("login/", auth_views.login_view, name="auth-login"),
    path("refresh/", auth_views.refresh_view, name="auth-refresh"),
    path("me/", auth_views.me_view, name="auth-me"),
]
