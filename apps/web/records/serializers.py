"""DRF serializers for records API."""
from rest_framework import serializers
from .models import GratifikasiRecord, AuditLog


class GratifikasiRecordSerializer(serializers.ModelSerializer):
    ai_result = serializers.SerializerMethodField()

    class Meta:
        model = GratifikasiRecord
        fields = [
            "id", "text", "value_estimation",
            "relationship", "context", "country", "regulatory_framework",
            "status", "ai_label", "ai_confidence", "ai_source", "ai_result",
            "final_label", "approved_by", "submitted_by",
            "created_at", "updated_at",
        ]
        read_only_fields = [
            "status", "ai_label", "ai_confidence", "ai_source", "ai_result",
            "final_label", "approved_by", "submitted_by", "created_at", "updated_at",
        ]

    def get_ai_result(self, obj):
        return obj.ai_result_json


class ApprovalSerializer(serializers.Serializer):
    final_label = serializers.ChoiceField(
        choices=["Milik Negara", "Bukan Milik Negara"]
    )
    note = serializers.CharField(required=False, allow_blank=True)


class AuditLogSerializer(serializers.ModelSerializer):
    timestamp = serializers.DateTimeField(source="created_at")

    class Meta:
        model = AuditLog
        fields = ["id", "record_id", "action", "actor", "note", "timestamp"]
