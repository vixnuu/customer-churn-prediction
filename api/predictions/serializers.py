
from rest_framework import serializers


class CustomerPredictionSerializer(serializers.Serializer):

    # --------------------------------------------------------
    # NUMERICAL FEATURES
    # --------------------------------------------------------

    SeniorCitizen = serializers.IntegerField(
        min_value=0,
        max_value=1
    )

    tenure = serializers.IntegerField(
        min_value=0
    )

    MonthlyCharges = serializers.FloatField(
        min_value=0
    )

    TotalCharges = serializers.FloatField(
        min_value=0
    )


    # --------------------------------------------------------
    # CATEGORICAL FEATURES
    # --------------------------------------------------------

    gender = serializers.ChoiceField(
        choices=["Female", "Male"]
    )

    Partner = serializers.ChoiceField(
        choices=["Yes", "No"]
    )

    Dependents = serializers.ChoiceField(
        choices=["Yes", "No"]
    )

    PhoneService = serializers.ChoiceField(
        choices=["Yes", "No"]
    )

    MultipleLines = serializers.ChoiceField(
        choices=[
            "Yes",
            "No",
            "No phone service"
        ]
    )

    InternetService = serializers.ChoiceField(
        choices=[
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    OnlineSecurity = serializers.ChoiceField(
        choices=[
            "Yes",
            "No",
            "No internet service"
        ]
    )

    OnlineBackup = serializers.ChoiceField(
        choices=[
            "Yes",
            "No",
            "No internet service"
        ]
    )

    DeviceProtection = serializers.ChoiceField(
        choices=[
            "Yes",
            "No",
            "No internet service"
        ]
    )

    TechSupport = serializers.ChoiceField(
        choices=[
            "Yes",
            "No",
            "No internet service"
        ]
    )

    StreamingTV = serializers.ChoiceField(
        choices=[
            "Yes",
            "No",
            "No internet service"
        ]
    )

    StreamingMovies = serializers.ChoiceField(
        choices=[
            "Yes",
            "No",
            "No internet service"
        ]
    )

    Contract = serializers.ChoiceField(
        choices=[
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    PaperlessBilling = serializers.ChoiceField(
        choices=["Yes", "No"]
    )

    PaymentMethod = serializers.ChoiceField(
        choices=[
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )
