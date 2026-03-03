"""Django admin configuration for records."""
from django.contrib import admin
from .models import GratifikasiRecord, AuditLog


@admin.register(GratifikasiRecord)
class GratifikasiRecordAdmin(admin.ModelAdmin):
    list_display = ["id", "status", "ai_label", "ai_confidence", "final_label", "created_at"]
    list_filter = ["status", "ai_label", "final_label"]
    search_fields = ["text"]
    readonly_fields = ["created_at", "updated_at", "ai_label", "ai_confidence", "ai_source"]
    ordering = ["-created_at"]


@admin.register(AuditLog)
class AuditLogAdmin(admin.ModelAdmin):
    list_display = ["id", "record", "action", "actor", "created_at"]
    list_filter = ["action"]
    readonly_fields = ["created_at"]
    ordering = ["-created_at"]
