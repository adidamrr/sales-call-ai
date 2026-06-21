import os

import requests
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


st.set_page_config(page_title="Sales Call AI", layout="wide")
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Sales Call AI")


def api_url(path):
    return f"{BACKEND_URL.rstrip('/')}{path}"


def card(title, text):
    st.header(title)
    st.caption(text)
    st.divider()


def status_badge(status):
    return status or "unknown"


def show_response(response):
    if response.ok:
        data = response.json()
        st.success("Готово")
        st.json(data)
        return data

    st.error(f"Ошибка запроса: {response.status_code}")
    try:
        st.json(response.json())
    except ValueError:
        st.write(response.text)
    return None


def get_managers():
    try:
        response = requests.get(api_url("/managers"), timeout=10)
        if response.ok:
            return response.json()
    except requests.RequestException:
        pass
    return []


def manager_display_name(manager):
    return f"{manager['name']} — {manager['department']} отдел — ID: {manager['id']}"


def create_manager_page():
    card(
        "Создание менеджера",
        "Добавьте менеджера, чтобы потом привязать к нему загруженный звонок.",
    )

    with st.form("create_manager_form"):
        manager_name = st.text_input("Имя менеджера")
        manager_department = st.text_input("Отдел", value="Sales")
        submitted = st.form_submit_button("Создать менеджера")

    if submitted:
        if not manager_name.strip():
            st.warning("Введите имя менеджера.")
            return

        response = requests.post(
            api_url("/managers"),
            json={
                "name": manager_name.strip(),
                "department": manager_department.strip() or "Sales",
            },
            timeout=20,
        )
        data = show_response(response)
        if data:
            st.success(f"Создан manager_id: {data['id']}")

    managers = get_managers()
    if managers:
        st.subheader("Текущие менеджеры")
        st.dataframe(managers, use_container_width=True)


def upload_call_page():
    card(
        "Загрузка звонка",
        (
            "Выберите менеджера и загрузите аудиофайл звонка. После загрузки звонок "
            "появится в списке и будет готов к транскрибации или анализу."
        ),
    )

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Менеджер")
        managers = get_managers()
        manager_options = {
            manager_display_name(manager): manager["id"]
            for manager in managers
        }

        selected_manager = None
        selected_manager_id = None

        if manager_options:
            selected_manager = st.selectbox(
                "Выберите существующего менеджера",
                list(manager_options),
            )
            selected_manager_id = manager_options[selected_manager]
        else:
            st.info("Пока нет менеджеров. Сначала создайте менеджера на первой странице.")

    with right:
        st.subheader("Аудиофайл")
        st.caption("Поддерживаемые форматы: MP3, WAV, M4A")
        audio_file = st.file_uploader("Загрузите файл", type=["mp3", "wav", "m4a"])

        if st.button("Загрузить звонок", type="primary"):
            if selected_manager_id is None:
                st.warning("Сначала выберите или создайте менеджера.")
                return

            if audio_file is None:
                st.warning("Выберите аудиофайл.")
                return

            files = {
                "file": (
                    audio_file.name,
                    audio_file.getvalue(),
                    audio_file.type or "application/octet-stream",
                )
            }
            response = requests.post(
                api_url("/calls/upload"),
                data={"manager_id": int(selected_manager_id)},
                files=files,
                timeout=60,
            )
            data = response.json() if response.ok else None
            if response.ok and data:
                st.success("Звонок загружен")
                st.write(f"Call ID: {data.get('call_id')}")
                st.write(f"Status: {status_badge(data.get('status'))}")
                st.write(f"Manager: {selected_manager}")
            else:
                show_response(response)


def calls_list_page():
    card(
        "Список звонков",
        "Смотрите загруженные звонки, их статусы и пути к аудиофайлам.",
    )

    response = requests.get(api_url("/calls"), timeout=20)
    if not response.ok:
        show_response(response)
        return

    calls = response.json()
    if not calls:
        st.info("Пока нет звонков.")
        return

    rows = [
        {
            "id": call.get("id"),
            "manager_id": call.get("manager_id"),
            "status": call.get("status"),
            "audio_path": call.get("audio_path"),
            "created_at": call.get("created_at"),
        }
        for call in calls
    ]
    st.dataframe(rows, use_container_width=True)


def analyze_page():
    card(
        "Анализ",
        "Запустите baseline-анализ или мультиагентный LangGraph-анализ для выбранного звонка.",
    )

    call_id = st.number_input("Call ID", min_value=1, value=1, step=1)
    method = st.selectbox(
        "Метод анализа",
        ["Baseline-анализ", "Мультиагентный LangGraph-анализ"],
    )
    endpoint_by_method = {
        "Baseline-анализ": ("analyze-basic", 180, "Выполняется baseline-анализ..."),
        "Мультиагентный LangGraph-анализ": (
            "analyze-agents",
            300,
            "Выполняется мультиагентный анализ...",
        ),
    }
    endpoint, timeout, spinner_text = endpoint_by_method[method]

    if st.button("Запустить анализ", type="primary", use_container_width=True):
        with st.spinner(spinner_text):
            response = requests.post(
                api_url(f"/calls/{int(call_id)}/{endpoint}"),
                timeout=timeout,
            )
        if response.status_code == 404:
            st.info(
                "Звонок с таким Call ID не найден или для него еще нет транскрипта. "
                "Проверьте Call ID и наличие транскрипта."
            )
            return

        show_response(response)


def show_list(title, items):
    st.subheader(title)
    if not items:
        st.write("Нет данных")
        return

    for item in items:
        st.markdown(f"- {item}")


def report_page():
    card(
        "Отчёт",
        "Получите сохранённый результат анализа: summary, оценку, рекомендации и найденные возражения.",
    )

    call_id = st.number_input("Call ID", min_value=1, value=1, step=1)
    if not st.button("Получить отчёт", type="primary"):
        return

    response = requests.get(api_url(f"/calls/{int(call_id)}/report"), timeout=30)
    if not response.ok:
        if response.status_code == 404:
            st.info(
                "Отчёт для этого звонка ещё не создан. "
                "Сначала запустите анализ на странице «Анализ»."
            )
            return

        show_response(response)
        return

    report = response.json()

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Summary")
        st.write(report.get("summary", ""))
        st.write(f"Call result: {status_badge(report.get('call_result', 'unknown'))}")

    with right:
        st.metric("Итоговая оценка", report.get("total_score", 0))

    show_list("Strengths", report.get("strengths", []))
    show_list("Weaknesses", report.get("weaknesses", []))
    show_list("Recommendations", report.get("recommendations", []))

    st.subheader("Objections")
    objections = report.get("objections", [])
    if objections:
        st.dataframe(objections, use_container_width=True)
    else:
        st.write("Нет возражений")

    st.subheader("Script compliance")
    script_compliance = report.get("script_compliance", [])
    if script_compliance:
        st.dataframe(script_compliance, use_container_width=True)
    else:
        st.write("Нет данных по скрипту")

    with st.expander("Raw report JSON"):
        st.json(report)


pages = {
    "Создание менеджера": create_manager_page,
    "Загрузка звонка": upload_call_page,
    "Список звонков": calls_list_page,
    "Анализ": analyze_page,
    "Отчёт": report_page,
}


selected_page = st.sidebar.radio("Навигация", list(pages))
pages[selected_page]()
