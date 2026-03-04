"""Authentication views — JWT login, me, user management."""
import logging
import httpx
import os
from django.contrib.auth import get_user_model
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
    qs = GratifikasiRecord.objects.all()
    total = qs.count()
    by_status = dict(
        qs.values_list("status").annotate(n=Count("id")).values_list("status", "n")
    )
    by_label = dict(
        qs.exclude(final_label=None)
        .values_list("final_label")
        .annotate(n=Count("id"))
        .values_list("final_label", "n")
    )
    by_ai_source = dict(
        qs.exclude(ai_source=None)
        .values_list("ai_source")
        .annotate(n=Count("id"))
        .values_list("ai_source", "n")
    )
    avg_confidence = (
        qs.exclude(ai_confidence=None)
        .values_list("ai_confidence", flat=True)
    )
    avg_conf_value = (
        round(sum(avg_confidence) / len(avg_confidence), 4)
        if avg_confidence
        else None
    )

    return Response({
        "total": total,
        "by_status": by_status,
        "by_label": by_label,
        "by_ai_source": by_ai_source,
        "avg_ai_confidence": avg_conf_value,
        "pending_review": by_status.get(RecordStatus.WAITING_APPROVAL, 0),
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
