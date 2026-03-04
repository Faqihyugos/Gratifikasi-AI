"""URL configuration for records app."""
from django.urls import path
from . import views
from . import auth_views

urlpatterns = [
    # Records
    path("records/", views.RecordListCreateView.as_view(), name="record-list-create"),
    path("records/<int:pk>/", views.RecordDetailView.as_view(), name="record-detail"),
    path("records/<int:pk>/approve/", views.approve_record, name="record-approve"),
    path("records/<int:pk>/audit/", views.record_audit_log, name="record-audit"),

    # Analytics
    path("analytics/", auth_views.analytics_view, name="analytics"),

    # Model info / retraining
    path("model-info/", auth_views.model_info_view, name="model-info"),
    path("model-info/retrain/", auth_views.retrain_view, name="model-retrain"),

    # User management (admin)
    path("users/", auth_views.users_list, name="users-list"),
    path("users/<int:pk>/", auth_views.user_detail, name="user-detail"),
]
