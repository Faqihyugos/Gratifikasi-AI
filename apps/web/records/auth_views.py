"""Authentication views — JWT login, me, user management."""
import logging
import httpx
import os
from django.contrib.auth import get_user_model
from django.db import models
from django.db.models import Count, Q
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated, IsAdminUser
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken

from .models import GratifikasiRecord, RecordStatus

User = get_user_model()
logger = logging.getLogger(__name__)

AI_SERVICE_URL = os.environ.get("AI_SERVICE_URL", "http://localhost:8001")

GROUP_ROLE_MAP = {
    "Compliance Officer": "compliance_officer",
    "Supervisor": "supervisor",
    "Auditor": "auditor",
    "ML Ops": "ml_ops",
}


def get_user_role(user) -> str:
    if user.is_superuser:
        return "admin"
    groups = user.groups.values_list("name", flat=True)
    for group_name, role in GROUP_ROLE_MAP.items():
        if group_name in groups:
            return role
    return "employee"


def user_to_dict(user) -> dict:
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "role": get_user_role(user),
        "is_active": user.is_active,
        "date_joined": user.date_joined.isoformat(),
    }


@api_view(["POST"])
@permission_classes([AllowAny])
def login_view(request):
    """Authenticate and return JWT access token + user info."""
    username = request.data.get("username", "").strip()
    password = request.data.get("password", "")

    if not username or not password:
        return Response(
            {"detail": "Username and password are required."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    from django.contrib.auth import authenticate
    user = authenticate(request, username=username, password=password)
    if user is None:
        return Response(
            {"detail": "Invalid credentials."},
            status=status.HTTP_401_UNAUTHORIZED,
        )
    if not user.is_active:
        return Response(
            {"detail": "Account is disabled."},
            status=status.HTTP_403_FORBIDDEN,
        )

    refresh = RefreshToken.for_user(user)
    refresh["role"] = get_user_role(user)
    refresh["username"] = user.username
    refresh["email"] = user.email

    return Response({
        "token": str(refresh.access_token),
        "refresh": str(refresh),
        "user": user_to_dict(user),
    })


@api_view(["POST"])
@permission_classes([AllowAny])
def refresh_view(request):
    """Return a new access token given a valid refresh token."""
    refresh_token = request.data.get("refresh")
    if not refresh_token:
        return Response({"detail": "Refresh token required."}, status=status.HTTP_400_BAD_REQUEST)
    try:
        refresh = RefreshToken(refresh_token)
        return Response({"token": str(refresh.access_token)})
    except Exception as exc:
        return Response({"detail": str(exc)}, status=status.HTTP_401_UNAUTHORIZED)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def me_view(request):
    """Return current authenticated user info."""
    return Response(user_to_dict(request.user))


# ---------------------------------------------------------------------------
# Users management (admin only)
# ---------------------------------------------------------------------------

@api_view(["GET"])
@permission_classes([IsAdminUser])
def users_list(request):
    """List all users."""
    users = User.objects.all().prefetch_related("groups").order_by("id")
    return Response([user_to_dict(u) for u in users])


@api_view(["PATCH"])
@permission_classes([IsAdminUser])
def user_detail(request, pk: int):
    """Update a user's role (group) or active status."""
    try:
        user = User.objects.prefetch_related("groups").get(pk=pk)
    except User.DoesNotExist:
        return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)

    if "is_active" in request.data:
        user.is_active = bool(request.data["is_active"])

    if "role" in request.data:
        role = request.data["role"]
        from django.contrib.auth.models import Group
        user.groups.clear()
        reverse_map = {v: k for k, v in GROUP_ROLE_MAP.items()}
        if role == "admin":
            user.is_superuser = True
            user.is_staff = True
        elif role in reverse_map:
            user.is_superuser = False
            user.is_staff = False
            group, _ = Group.objects.get_or_create(name=reverse_map[role])
            user.groups.add(group)
        else:
            user.is_superuser = False
            user.is_staff = False

    user.save()
    return Response(user_to_dict(user))


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def analytics_view(request):
    """Return aggregate statistics for the dashboard."""
    from django.utils import timezone
    from datetime import timedelta

    qs = GratifikasiRecord.objects.all()
    total = qs.count()

    # Label distribution
    mn_count = qs.filter(final_label="Milik Negara").count()
    bmn_count = qs.filter(final_label="Bukan Milik Negara").count()
    labelled = mn_count + bmn_count

    milik_negara_pct = (mn_count / labelled * 100) if labelled else 0
    bukan_milik_negara_pct = (bmn_count / labelled * 100) if labelled else 0

    # AI source distribution
    sim_count = qs.filter(ai_source="similarity").count()
    clf_count = qs.filter(ai_source="classifier").count()
    sourced = sim_count + clf_count
    similarity_based_pct = (sim_count / sourced * 100) if sourced else 0
    classifier_based_pct = (clf_count / sourced * 100) if sourced else 0

    # Override rate: approved with final_label != ai_label
    approved_qs = qs.filter(status=RecordStatus.APPROVED)
    approved_count = approved_qs.count()
    override_count = approved_qs.exclude(final_label=models.F("ai_label")).count()
    override_rate = (override_count / approved_count) if approved_count else 0

    # Avg approval time (hours): from created_at to updated_at for APPROVED records
    from django.db.models import ExpressionWrapper, DurationField
    durations = (
        approved_qs.annotate(
            duration=ExpressionWrapper(
                models.F("updated_at") - models.F("created_at"),
                output_field=DurationField(),
            )
        )
        .values_list("duration", flat=True)
    )
    if durations:
        avg_seconds = sum(d.total_seconds() for d in durations if d) / len(durations)
        avg_approval_time_hours = round(avg_seconds / 3600, 2)
    else:
        avg_approval_time_hours = 0

    # Submissions by month (last 6 months)
    from django.db.models.functions import TruncMonth
    monthly = (
        qs.annotate(month=TruncMonth("created_at"))
        .values("month")
        .annotate(count=Count("id"))
        .order_by("month")
    )
    submissions_by_month = [
        {"month": m["month"].strftime("%b %Y"), "count": m["count"]}
        for m in monthly
        if m["month"]
    ]

    # Label distribution list
    label_distribution = [
        {"label": "Milik Negara", "count": mn_count},
        {"label": "Bukan Milik Negara", "count": bmn_count},
    ]

    return Response({
        "total_submissions": total,
        "milik_negara_pct": round(milik_negara_pct, 2),
        "bukan_milik_negara_pct": round(bukan_milik_negara_pct, 2),
        "similarity_based_pct": round(similarity_based_pct, 2),
        "classifier_based_pct": round(classifier_based_pct, 2),
        "override_rate": round(override_rate, 4),
        "avg_approval_time_hours": avg_approval_time_hours,
        "submissions_by_month": submissions_by_month,
        "label_distribution": label_distribution,
        "pending_review": qs.filter(status=RecordStatus.WAITING_APPROVAL).count(),
        "avg_ai_confidence": round(
            sum(v for v in qs.exclude(ai_confidence=None).values_list("ai_confidence", flat=True)) /
            max(qs.exclude(ai_confidence=None).count(), 1),
            4,
        ),
    })


# ---------------------------------------------------------------------------
# Model info (proxy to AI service)
# ---------------------------------------------------------------------------

@api_view(["GET"])
@permission_classes([IsAuthenticated])
def model_info_view(request):
    """Proxy to AI service /model endpoint."""
    try:
        resp = httpx.get(f"{AI_SERVICE_URL}/model", timeout=5.0)
        return Response(resp.json(), status=resp.status_code)
    except Exception as exc:
        logger.warning("AI service unavailable: %s", exc)
        return Response({"detail": "AI service unavailable."}, status=status.HTTP_503_SERVICE_UNAVAILABLE)


@api_view(["POST"])
@permission_classes([IsAdminUser])
def retrain_view(request):
    """Trigger a retraining run (admin only)."""
    from .tasks import run_ai_task  # noqa: F401 — just to ensure celery is wired
    # In production, trigger via a dedicated Celery task or CI job.
    return Response({"message": "Retraining pipeline is profile-gated. Run: docker compose run --rm trainer"})
