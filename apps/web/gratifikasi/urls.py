from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("records.urls")),
    path("auth/", include("records.auth_urls")),
]
