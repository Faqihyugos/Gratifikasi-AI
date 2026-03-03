"""Database models for Gratifikasi records."""
from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()


class RecordStatus(models.TextChoices):
    PENDING = "PENDING", "Pending"
    PROCESSING = "PROCESSING", "Processing"
    WAITING_APPROVAL = "WAITING_APPROVAL", "Waiting Approval"
    APPROVED = "APPROVED", "Approved"
    REJECTED = "REJECTED", "Rejected"


class AiSource(models.TextChoices):
    SIMILARITY = "similarity", "Similarity"
    CLASSIFIER = "classifier", "Classifier"
    UNKNOWN = "unknown", "Unknown"


class GratifikasiRecord(models.Model):
    """Main record for gratification cases."""
    text = models.TextField(help_text="Description of the gratification case")
    value_estimation = models.DecimalField(
        max_digits=15, decimal_places=2, null=True, blank=True
    )
    status = models.CharField(
        max_length=30, choices=RecordStatus.choices, default=RecordStatus.PENDING
    )
    ai_label = models.CharField(max_length=50, null=True, blank=True)
    ai_confidence = models.FloatField(null=True, blank=True)
    ai_source = models.CharField(
        max_length=20, choices=AiSource.choices, null=True, blank=True
    )
    final_label = models.CharField(max_length=50, null=True, blank=True)
    approved_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="approved_records",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"GratifikasiRecord({self.id}, {self.status})"


class AuditLog(models.Model):
    """Audit log for record actions."""
    record = models.ForeignKey(
        GratifikasiRecord,
        on_delete=models.CASCADE,
        related_name="audit_logs",
    )
    action = models.CharField(max_length=100)
    actor = models.CharField(max_length=150)
    note = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"AuditLog({self.record_id}, {self.action}, {self.actor})"
