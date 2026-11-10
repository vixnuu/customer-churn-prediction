
from unittest.mock import patch

from django.urls import reverse

from rest_framework import status
from rest_framework.test import APITestCase


class CustomerPredictionAPITests(APITestCase):

    valid_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.50,
        "TotalCharges": 1074.00
    }


    def test_health_endpoint(self):

        response = self.client.get(
            reverse("api-health")
        )

        self.assertEqual(
            response.status_code,
            status.HTTP_200_OK
        )

        self.assertEqual(
            response.data["status"],
            "healthy"
        )


    @patch("predictions.views.predict_customer")
    def test_valid_prediction(self, mock_predict):

        mock_predict.return_value = {
            "churn_prediction": 1,
            "churn_label": "Yes",
            "churn_score": 0.86,
            "decision_threshold": 0.547
        }

        response = self.client.post(
            reverse("churn-predict"),
            self.valid_customer,
            format="json"
        )

        self.assertEqual(
            response.status_code,
            status.HTTP_200_OK
        )

        self.assertEqual(
            response.data["churn_label"],
            "Yes"
        )

        mock_predict.assert_called_once()


    @patch("predictions.views.predict_customer")
    def test_missing_feature_returns_400(self, mock_predict):

        invalid_customer = self.valid_customer.copy()

        invalid_customer.pop("tenure")

        response = self.client.post(
            reverse("churn-predict"),
            invalid_customer,
            format="json"
        )

        self.assertEqual(
            response.status_code,
            status.HTTP_400_BAD_REQUEST
        )

        mock_predict.assert_not_called()


    @patch("predictions.views.predict_customer")
    def test_invalid_category_returns_400(self, mock_predict):

        invalid_customer = self.valid_customer.copy()

        invalid_customer["Contract"] = "Invalid Contract"

        response = self.client.post(
            reverse("churn-predict"),
            invalid_customer,
            format="json"
        )

        self.assertEqual(
            response.status_code,
            status.HTTP_400_BAD_REQUEST
        )

        mock_predict.assert_not_called()
