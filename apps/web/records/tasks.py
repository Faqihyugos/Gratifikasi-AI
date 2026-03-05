"""Celery tasks for async AI processing."""
import logging
import httpx
from celery import shared_task
from django.conf import settings

from .models import GratifikasiRecord, AuditLog, RecordStatus, AiSource

logger = logging.getLogger(__name__)

AI_TIMEOUT = 30  # seconds


@shared_task(
    bind=True,
    max_retries=3,
    default_retry_delay=10,
    name="records.run_ai_task",
)
def run_ai_task(self, record_id: int) -> None:
    """Call AI service to predict label for a record."""
    try:
        record = GratifikasiRecord.objects.get(id=record_id)
    except GratifikasiRecord.DoesNotExist:
        logger.error("Record %s not found", record_id)
        return

    ai_url = settings.AI_SERVICE_URL
    try:
        with httpx.Client(timeout=AI_TIMEOUT) as client:
            resp = client.post(
                f"{ai_url}/predict",
                json={
                    "text": record.text,
                    "top_k": settings.AI_PREDICT_TOP_K,
                    "similarity_threshold": settings.AI_SIMILARITY_THRESHOLD,
                },
            )
        resp.raise_for_status()
        data = resp.json()

        record.ai_label = data.get("label")
        record.ai_confidence = data.get("confidence")
        source = data.get("source", "unknown")
        valid_sources = [choice[0] for choice in AiSource.choices]
        record.ai_source = source if source in valid_sources else AiSource.UNKNOWN
        record.ai_result_json = data  # store full AI response for frontend
        record.status = RecordStatus.WAITING_APPROVAL
        record.save()

        AuditLog.objects.create(
            record=record,
            action="AI_PROCESSED",
            actor="ai_service",
            note=(
                f"AI label: {record.ai_label} "
                f"(confidence: {record.ai_confidence:.4f}, source: {source})"
            ),
        )
        logger.info("AI processed record %s: %s", record_id, record.ai_label)

    except httpx.HTTPError as exc:
        logger.warning("AI service error for record %s: %s", record_id, exc)
        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            record.status = RecordStatus.WAITING_APPROVAL
            record.ai_source = AiSource.UNKNOWN
            record.save()
            AuditLog.objects.create(
                record=record,
                action="AI_FAILED",
                actor="system",
                note=f"AI service unavailable: {exc}",
            )


@shared_task(
    bind=True,
    max_retries=3,
    default_retry_delay=10,
    name="records.upsert_to_qdrant_task",
)
def upsert_to_qdrant_task(self, record_id: int) -> None:
    """Upsert approved record into Qdrant via AI service."""
    try:
        record = GratifikasiRecord.objects.get(id=record_id)
    except GratifikasiRecord.DoesNotExist:
        logger.error("Record %s not found", record_id)
        return

    if not record.final_label:
        logger.warning("Record %s has no final_label, skipping upsert", record_id)
        return

    ai_url = settings.AI_SERVICE_URL
    try:
        with httpx.Client(timeout=AI_TIMEOUT) as client:
            resp = client.post(
                f"{ai_url}/cases/upsert",
                json={
                    "record_id": record.id,
                    "text": record.text,
                    "final_label": record.final_label,
                    "value_estimation": (
                        float(record.value_estimation)
                        if record.value_estimation else None
                    ),
                    "created_at": record.created_at.isoformat(),
                },
            )
        resp.raise_for_status()
        logger.info("Upserted record %s to Qdrant", record_id)

    except httpx.HTTPError as exc:
        logger.warning("Qdrant upsert error for record %s: %s", record_id, exc)
        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            logger.error("Failed to upsert record %s after retries", record_id)


@shared_task(name="records.data_drift_check")
def data_drift_check() -> None:
    """
    Periodic task: check data volume and trigger training if threshold met.
    Run daily at 02:00 via Celery beat.

    NOTE: If approved_count >= NEW_LABELS_THRESHOLD, this task logs a warning
    and the training must be triggered manually with:
      docker compose run --rm trainer
    Or via a webhook/management command from your CI/CD system.
    """
    threshold = 200

    approved_count = GratifikasiRecord.objects.filter(
        final_label__isnull=False
    ).count()

    logger.info("Data drift check: %d approved records", approved_count)

    if approved_count >= threshold:
        logger.warning(
            "Training threshold reached (%d >= %d). "
            "Trigger training manually: docker compose run --rm trainer",
            approved_count,
            threshold,
        )
        # Hook: place your CD trigger here (e.g. call an external webhook)
