"""DRF serializers for records API."""
from rest_framework import serializers
from .models import GratifikasiRecord, AuditLog


class GratifikasiRecordSerializer(serializers.ModelSerializer):
    class Meta:
        model = GratifikasiRecord
        fields = [
            "id", "text", "value_estimation", "status",
            "ai_label", "ai_confidence", "ai_source",
            "final_label", "approved_by",
            "created_at", "updated_at",
        ]
        read_only_fields = [
            "status", "ai_label", "ai_confidence", "ai_source",
            "final_label", "approved_by", "created_at", "updated_at",
        ]


class ApprovalSerializer(serializers.Serializer):
    final_label = serializers.ChoiceField(
        choices=["Milik Negara", "Bukan Milik Negara"]
    )
    note = serializers.CharField(required=False, allow_blank=True)


class AuditLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = AuditLog
        fields = ["id", "record_id", "action", "actor", "note", "created_at"]
