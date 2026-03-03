"""URL configuration for records app."""
from django.urls import path
from . import views

urlpatterns = [
    path("records/", views.RecordListCreateView.as_view(), name="record-list-create"),
    path("records/<int:pk>/", views.RecordDetailView.as_view(), name="record-detail"),
    path("records/<int:pk>/approve/", views.approve_record, name="record-approve"),
    path("records/<int:pk>/audit/", views.record_audit_log, name="record-audit"),
]
