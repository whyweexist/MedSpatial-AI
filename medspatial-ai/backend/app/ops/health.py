from app.config import settings
from app.core.hardware_manager import get_hardware_manager


def health_snapshot() -> dict:
    return {
        "status": "healthy",
        "architecture": "AWM-JGEM",
        "profile": settings.PROFILE,
        "remote_providers_enabled": settings.ENABLE_REMOTE_PROVIDERS,
        "hardware": get_hardware_manager().get_status_dict(),
    }
