
import logging

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework import status

from .serializers import CustomerPredictionSerializer
from .services import predict_customer


logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# 1. HEALTH CHECK ENDPOINT
# ------------------------------------------------------------

class HealthCheckView(APIView):

    permission_classes = [AllowAny]

    def get(self, request):

        return Response({
            "status": "healthy",
            "service": "Customer Churn Prediction API"
        })


# ------------------------------------------------------------
# 2. CUSTOMER CHURN PREDICTION ENDPOINT
# ------------------------------------------------------------

class ChurnPredictionView(APIView):

    permission_classes = [AllowAny]

    def post(self, request):

        # Validate incoming JSON
        serializer = CustomerPredictionSerializer(
            data=request.data
        )

        if not serializer.is_valid():

            return Response(
                {
                    "errors": serializer.errors
                },
                status=status.HTTP_400_BAD_REQUEST
            )

        try:

            # Send validated customer information
            # to the saved ML prediction pipeline.

            result = predict_customer(
                serializer.validated_data
            )

            return Response(
                result,
                status=status.HTTP_200_OK
            )

        except FileNotFoundError:

            logger.exception("ML model artifacts are missing.")

            return Response(
                {
                    "detail": (
                        "Prediction model is unavailable. "
                        "Train and save the model first."
                    )
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        except Exception:

            logger.exception("Customer prediction failed.")

            return Response(
                {
                    "detail": "Prediction could not be completed."
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
