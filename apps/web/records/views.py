"""API views for Gratifikasi records."""
import logging
from rest_framework import generics, status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from django.shortcuts import get_object_or_404

from .models import GratifikasiRecord, AuditLog, RecordStatus
from .serializers import GratifikasiRecordSerializer, ApprovalSerializer, AuditLogSerializer
from .tasks import run_ai_task, upsert_to_qdrant_task

logger = logging.getLogger(__name__)


class RecordListCreateView(generics.ListCreateAPIView):
    """List all records or submit a new one."""
    serializer_class = GratifikasiRecordSerializer
    permission_classes = [AllowAny]

    def get_queryset(self):
        qs = GratifikasiRecord.objects.all()
        if self.request.query_params.get("mine") and self.request.user.is_authenticated:
            qs = qs.filter(submitted_by=self.request.user)
        status_filter = self.request.query_params.get("status")
        if status_filter:
            qs = qs.filter(status=status_filter)
        return qs

    def perform_create(self, serializer):
        user = self.request.user if self.request.user.is_authenticated else None
        record = serializer.save(status=RecordStatus.PROCESSING, submitted_by=user)
        AuditLog.objects.create(
            record=record,
            action="SUBMITTED",
            actor=str(self.request.user) if self.request.user.is_authenticated else "anonymous",
            note="Record submitted",
        )
        run_ai_task.delay(record.id)
        logger.info("Submitted record %s, queued AI task", record.id)


class RecordDetailView(generics.RetrieveAPIView):
    """Retrieve a single record."""
    queryset = GratifikasiRecord.objects.all()
    serializer_class = GratifikasiRecordSerializer
    permission_classes = [AllowAny]


@api_view(["POST"])
@permission_classes([AllowAny])
def approve_record(request, pk: int):
    """Approve or reject a record with a final label."""
    record = get_object_or_404(GratifikasiRecord, pk=pk)
    if record.status not in [RecordStatus.WAITING_APPROVAL]:
        return Response(
            {"detail": f"Record is in status {record.status}, cannot approve."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    serializer = ApprovalSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)

    final_label = serializer.validated_data["final_label"]
    note = serializer.validated_data.get("note", "")

    record.final_label = final_label
    record.status = RecordStatus.APPROVED
    record.approved_by = request.user if request.user.is_authenticated else None
    record.save()

    actor = str(request.user) if request.user.is_authenticated else "anonymous"
    AuditLog.objects.create(
        record=record,
        action="APPROVED",
        actor=actor,
        note=note or f"Approved with label: {final_label}",
    )

    upsert_to_qdrant_task.delay(record.id)
    logger.info("Record %s approved with label %s", record.id, final_label)

    return Response(GratifikasiRecordSerializer(record).data)


@api_view(["GET"])
@permission_classes([AllowAny])
def record_audit_log(request, pk: int):
    """Get audit log for a record."""
    record = get_object_or_404(GratifikasiRecord, pk=pk)
    logs = AuditLog.objects.filter(record=record)
    return Response(AuditLogSerializer(logs, many=True).data)
