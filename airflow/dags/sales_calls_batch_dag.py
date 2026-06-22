import os
from datetime import datetime, timedelta

import requests
from airflow import DAG
from airflow.operators.python import PythonOperator


BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 30


def get_backend(path, params=None):
    response = requests.get(
        f"{BACKEND_URL}{path}",
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def post_backend(path):
    response = requests.post(
        f"{BACKEND_URL}{path}",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def get_uploaded_calls():
    calls = get_backend("/calls", params={"status": "uploaded"})
    return [call["id"] for call in calls]


def transcribe_uploaded_calls(**context):
    call_ids = context["ti"].xcom_pull(task_ids="get_uploaded_calls") or []
    started_call_ids = []

    for call_id in call_ids:
        post_backend(f"/calls/{call_id}/transcribe-async")
        started_call_ids.append(call_id)

    return started_call_ids


def get_transcribed_calls():
    calls = get_backend("/calls", params={"status": "transcribed"})
    return [call["id"] for call in calls]


def analyze_transcribed_calls(**context):
    call_ids = context["ti"].xcom_pull(task_ids="get_transcribed_calls") or []
    started_call_ids = []

    for call_id in call_ids:
        post_backend(f"/calls/{call_id}/analyze-agents-async")
        started_call_ids.append(call_id)

    return started_call_ids


default_args = {
    "owner": "sales-call-ai",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="sales_calls_batch_pipeline",
    default_args=default_args,
    description="Nightly batch processing for uploaded sales calls.",
    schedule_interval="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["sales-call-ai", "batch"],
) as dag:
    get_uploaded_calls_task = PythonOperator(
        task_id="get_uploaded_calls",
        python_callable=get_uploaded_calls,
    )

    transcribe_uploaded_calls_task = PythonOperator(
        task_id="transcribe_uploaded_calls",
        python_callable=transcribe_uploaded_calls,
    )

    get_transcribed_calls_task = PythonOperator(
        task_id="get_transcribed_calls",
        python_callable=get_transcribed_calls,
    )

    analyze_transcribed_calls_task = PythonOperator(
        task_id="analyze_transcribed_calls",
        python_callable=analyze_transcribed_calls,
    )

    (
        get_uploaded_calls_task
        >> transcribe_uploaded_calls_task
        >> get_transcribed_calls_task
        >> analyze_transcribed_calls_task
    )
