from urllib import response

import pandas as pd
from api.services.model_loader import load_risk_model, load_claim_model
from monitoring.logger import log_prediction

def predict_risk_result(data:dict):
    model, model_name, version = load_risk_model()

    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)[0]
    prediction_str = str(prediction)

    response = {
        "prediction": prediction_str
    }

    log_prediction(
        model_name=model_name,
        model_version=version,
        input_data=data,
        prediction=prediction_str
    )

    # Checking for probabilities attribute
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(input_df)[0]
            if hasattr(probabilities, "tolist"):
                response["probabilities"] = probabilities.tolist()
            else:
                response["probabilities"] = probabilities
        except Exception as ex:
            response["probabilities_error"] = str(ex)


    return response





def predict_claim_result(data:dict):
    model, model_name, version = load_claim_model()

    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)[0]
    prediction_str = str(prediction)

    response = {
        "prediction": prediction_str
    }

    log_prediction(
        model_name=model_name,
        model_version=version,
        input_data=data,
        prediction=prediction_str
    )

    # Checking for probabilities attribute
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(input_df)[0]
            if hasattr(probabilities, "tolist"):
                response["probabilities"] = probabilities.tolist()
            else:
                response["probabilities"] = probabilities
        except Exception as ex:
            response["probabilities_error"] = str(ex)

    return response