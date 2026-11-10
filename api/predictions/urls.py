from django.urls import path

from .views import (
    HealthCheckView,
    ChurnPredictionView
)


urlpatterns = [
    path(
        "health/",
        HealthCheckView.as_view(),
        name="api-health"
    ),

    path(
        "predict/",
        ChurnPredictionView.as_view(),
        name="churn-predict"
    )
]