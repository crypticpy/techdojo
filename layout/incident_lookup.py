# layout/incident_lookup.py

import streamlit as st
import pandas as pd
from datetime import datetime
import os
import uuid
import json
import threading
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import urllib.request
import ssl

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Azure ML endpoint URL
AZURE_ENDPOINT = "https://rg-aph-mlworkspace-snpred.southcentralus.inference.ml.azure.com/score"

# Add your API key here
API_KEY = "M9VvmEKDL0B9fs9CUhLCF8W5qvomgT7e"

# File paths
FEEDBACK_FILE = "feedback_data.csv"
PREDICTIONS_FILE = "predictions_log.csv"

# Lock for thread-safe file operations
file_lock = threading.Lock()

class IncidentData(BaseModel):
    incident_id: str = Field(..., description="Unique identifier for the incident")
    incident_number: Optional[str] = Field(None, description="Incident Number")
    contact_type: Optional[str] = Field(None, description="Contact Type")
    requested_for_title: Optional[str] = Field(None, description="Requested For Title")
    requested_for_department: Optional[str] = Field(None, description="Requested For Department")
    requested_for_location: Optional[str] = Field(None, description="Requested For Location")
    category: Optional[str] = Field(None, description="Category")
    sub_category: Optional[str] = Field(None, description="Sub Category")
    priority: Optional[str] = Field(None, description="Priority")
    description: Optional[str] = Field(None, description="Description")
    extract_product: Optional[str] = Field(None, description="Extract Product")
    summary: Optional[str] = Field(None, description="Summary")

def load_incident_data(incident_number: str) -> Optional[IncidentData]:
    """Loads incident data from a JSON file based on the incident number."""
    file_name = "incidents.json"
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", file_name)

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            incidents = json.load(f)
            incident = next((inc for inc in incidents if inc.get("incident_number") == incident_number), None)
            if incident:
                # Replace "null" string values with None
                incident = {k: (None if v == "null" else v) for k, v in incident.items()}
                return IncidentData(**incident)
    return None

def predict_assignment_group(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict assignment group using Azure ML endpoint.
    """
    def allowSelfSignedHttps(allowed):
        # bypass the server certificate verification on client side
        if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
            ssl._create_default_https_context = ssl._create_unverified_context

    allowSelfSignedHttps(True)  # this line is needed if you use self-signed certificate in your scoring service.

    input_data = {
        "input_data": {
            "columns": [
                "Contact Type", "Requested For Title", "Requested For Department",
                "Requested For Location", "Category", "Sub Category", "Priority",
                "Description", "extract_product", "summary"
            ],
            "index": [0],
            "data": [[data.get(field, "") for field in [
                "Contact Type", "Requested For Title", "Requested For Department",
                "Requested For Location", "Category", "Sub Category", "Priority",
                "Description", "extract_product", "summary"
            ]]]
        }
    }

    body = str.encode(json.dumps(input_data))

    if not API_KEY:
        raise Exception("A key should be provided to invoke the endpoint")

    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ API_KEY)}

    req = urllib.request.Request(AZURE_ENDPOINT, body, headers)

    try:
        response = urllib.request.urlopen(req)
        result = response.read()
        return json.loads(result.decode())
    except urllib.error.HTTPError as error:
        logging.error(f"The request failed with status code: {str(error.code)}")
        logging.error(error.info())
        logging.error(error.read().decode("utf8", 'ignore'))
        return None

def save_data(data: Dict[str, Any], filename: str) -> None:
    """
    Save data to CSV file in a thread-safe manner.
    """
    with file_lock:
        try:
            if not os.path.isfile(filename):
                pd.DataFrame([data]).to_csv(filename, index=False)
            else:
                pd.DataFrame([data]).to_csv(filename, mode='a', header=False, index=False)
        except Exception as e:
            logging.error(f"Error saving data to {filename}: {e}")

def save_feedback(original_data: Dict[str, Any], predicted_group: str, correct_group: str, feedback: str,
                  session_id: str) -> None:
    """
    Save feedback data.
    """
    feedback_data = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "predicted_group": predicted_group,
        "correct_group": correct_group,
        "feedback": feedback,
        **original_data
    }
    save_data(feedback_data, FEEDBACK_FILE)

def log_prediction(data: Dict[str, Any], prediction: Dict[str, Any], session_id: str) -> None:
    """
    Log prediction data.
    """
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "prediction": json.dumps(prediction),
        **data
    }
    save_data(log_data, PREDICTIONS_FILE)

def reset_form() -> None:
    """
    Reset the form by clearing session state.
    """
    for key in list(st.session_state.keys()):
        if key not in ['session_id', 'prediction_result']:
            del st.session_state[key]

def display_incident_lookup_page():
    st.title("ServiceNow Ticket Assignment Group Predictor")

    # Initialize session state
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    # Add a placeholder for the routing group at the top
    top_routing_group = st.empty()

    st.write("Please enter the Incident Number or fill in the ticket information:")

    # Input fields
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        incident_number = st.text_input("Incident Number", key="incident_number")
    with col2:
        pull_data = st.button("Pull Data")
    with col3:
        route_button = st.button("Route")

    # Initialize session state for form fields if not already present
    if 'form_data' not in st.session_state:
        st.session_state.form_data = {
            "contact_type": "",
            "requested_for_title": "",
            "requested_for_department": "",
            "requested_for_location": "",
            "category": "",
            "sub_category": "",
            "priority": "3 - Moderate",
            "description": "",
            "extract_product": "",
            "summary": ""
        }

    if pull_data and incident_number:
        incident_data = load_incident_data(incident_number)
        if incident_data:
            st.success(f"Data loaded for Incident Number: {incident_number}")
            # Update session state with loaded data
            for field, value in incident_data.dict().items():
                if field != "incident_id" and value is not None:
                    st.session_state.form_data[field] = value
        else:
            st.warning(f"No data found for Incident Number: {incident_number}")

    # Display form fields using session state
    contact_type = st.text_input("Contact Type", value=st.session_state.form_data["contact_type"], key="contact_type")
    requested_for_title = st.text_input("Requested For Title", value=st.session_state.form_data["requested_for_title"], key="requested_for_title")
    requested_for_department = st.text_input("Requested For Department", value=st.session_state.form_data["requested_for_department"], key="requested_for_department")
    requested_for_location = st.text_input("Requested For Location", value=st.session_state.form_data["requested_for_location"], key="requested_for_location")
    category = st.text_input("Category", value=st.session_state.form_data["category"], key="category")
    sub_category = st.text_input("Sub Category", value=st.session_state.form_data["sub_category"], key="sub_category")
    priority = st.selectbox("Priority", ["1 - Critical", "2 - High", "3 - Moderate", "4 - Low"], index=["1 - Critical", "2 - High", "3 - Moderate", "4 - Low"].index(st.session_state.form_data["priority"]), key="priority")
    description = st.text_area("Description", value=st.session_state.form_data["description"], key="description")
    extract_product = st.text_input("Extract Product", value=st.session_state.form_data["extract_product"], key="extract_product")
    summary = st.text_area("Summary", value=st.session_state.form_data["summary"], key="summary")

    # Add a field to display the routing group at the bottom
    bottom_routing_group = st.empty()

    if route_button or st.button("Predict Assignment Group"):
        ticket_data = {
            "Incident Number": incident_number,
            "Contact Type": contact_type,
            "Requested For Title": requested_for_title,
            "Requested For Department": requested_for_department,
            "Requested For Location": requested_for_location,
            "Category": category,
            "Sub Category": sub_category,
            "Priority": priority,
            "Description": description,
            "extract_product": extract_product,
            "summary": summary
        }

        with st.spinner('Predicting...'):
            prediction_result = predict_assignment_group(ticket_data)

        if prediction_result:
            st.session_state.prediction_result = prediction_result
            st.session_state.ticket_data = ticket_data

            # Handle string, list, and dictionary responses
            if isinstance(prediction_result, str):
                try:
                    predictions = json.loads(prediction_result)
                except json.JSONDecodeError:
                    predictions = {"Unknown": 1.0}
            elif isinstance(prediction_result, list):
                predictions = prediction_result[0] if prediction_result else {}
            elif isinstance(prediction_result, dict):
                predictions = prediction_result.get('output', {})
            else:
                predictions = {}

            if predictions:
                if isinstance(predictions, dict):
                    top_prediction = max(predictions, key=predictions.get)
                else:
                    top_prediction = str(predictions)
                routing_message = f"Routing Group: {top_prediction}"
                top_routing_group.success(routing_message)
                bottom_routing_group.success(routing_message)
            else:
                routing_message = "No prediction available"
                top_routing_group.warning(routing_message)
                bottom_routing_group.warning(routing_message)

            log_prediction(ticket_data, prediction_result, st.session_state.session_id)
        else:
            st.error("Error: Unable to get predictions. Please try again.")

        if 'prediction_result' in st.session_state and st.session_state.prediction_result:
            st.success("Predictions:")

            # Handle string, list, and dictionary responses
            prediction_result = st.session_state.prediction_result
            if isinstance(prediction_result, str):
                try:
                    predictions = json.loads(prediction_result)
                except json.JSONDecodeError:
                    predictions = {"Unknown": 1.0}
            elif isinstance(prediction_result, list):
                predictions = prediction_result[0] if prediction_result else {}
            elif isinstance(prediction_result, dict):
                predictions = prediction_result.get('output', {})
            else:
                predictions = {}

            if predictions:
                if isinstance(predictions, dict):
                    predicted_groups = []
                    for group, probability in predictions.items():
                        st.write(f"Assignment Group: {group}, Confidence: {probability:.2%}")
                        predicted_groups.append(group)
                    top_prediction = predicted_groups[0] if predicted_groups else "Unknown"
                else:
                    st.write(f"Assignment Group: {predictions}")
                    top_prediction = str(predictions)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Correct"):
                    save_feedback(st.session_state.ticket_data, top_prediction, top_prediction, "correct",
                                  st.session_state.session_id)
                    st.success("Feedback recorded. Thank you!")
                    reset_form()
                    st.rerun()
            with col2:
                if st.button("👎 Incorrect"):
                    st.session_state.feedback_incorrect = True
                    st.rerun()

            if st.session_state.get('feedback_incorrect', False):
                correct_group = st.text_input("Please enter the correct assignment group:")
                if st.button("Save and Reset"):
                    save_feedback(st.session_state.ticket_data, top_prediction, correct_group, "incorrect",
                                  st.session_state.session_id)
                    st.success("Feedback recorded. Thank you!")
                    reset_form()
                    st.rerun()
        else:
            st.warning("No predictions available.")

    st.button("Reset", on_click=reset_form)

    st.info("Note: Some fields are optional and can be left blank if the information is not available.")

    # Add a button to move to the troubleshooting interface
    if st.button("Move to Troubleshooting"):
        st.session_state.move_to_troubleshooting = True
        st.session_state.current_page = "Troubleshooting"
        st.rerun()
