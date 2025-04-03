import gradio as gr
import os
import time
from dotenv import load_dotenv

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- Configuration ---
load_dotenv()

# --- Constants ---
APP_TITLE = "PRODUCTIVITY PAL"
DEFAULT_MODEL = "gemini-1.5-flash-latest"
MAX_HISTORY_PAIRS = 5
POMODORO_WORK_MINS = 25
POMODORO_BREAK_MINS = 5

# --- LangChain System Prompt ---
SYSTEM_PROMPT = """You are 'Productivity Pal', a focused AI assistant for productivity tasks like task management, note-taking, motivation, and brainstorming productivity ideas. Politely decline requests outside this scope. Be friendly, concise, and use markdown formatting."""

# --- Chat History Store ---
# Simple in-memory store for session history
chat_history_store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    """Retrieves or creates chat history for a session."""
    if session_id not in chat_history_store:
        chat_history_store[session_id] = ChatMessageHistory()
    return chat_history_store[session_id]

# --- Core Chat Logic (LCEL) ---
def create_llm_chain(api_key):
    """Creates the LLM and the runnable chain with history."""
    llm = ChatGoogleGenerativeAI(
        model=DEFAULT_MODEL,
        google_api_key=api_key,
        temperature=0.6,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    runnable = prompt | llm
    chain_with_history = RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    return chain_with_history

# --- Gradio UI & Logic ---

def chatbot_response(api_key: str, user_input: str, chat_history_list: list):
    """Handles user input, runs the LLM chain, returns updated UI history."""
    session_id = "prod_pal_session" # Single session ID for this app

    if not api_key:
        yield "", chat_history_list + [(user_input, "⚠️ Error: API Key is missing.")]
        return
    if not user_input.strip():
         yield "", chat_history_list
         return

    try:
        # 1. Create the chain (includes LLM initialization)
        chain = create_llm_chain(api_key)

        # 2. Load current UI history into the LangChain history object
        lc_history = get_session_history(session_id)
        lc_history.clear()
        # Load only the last N pairs
        recent_gradio_history = chat_history_list[-(MAX_HISTORY_PAIRS):]
        for user_msg, ai_msg in recent_gradio_history:
            if user_msg: lc_history.add_user_message(user_msg)
            if ai_msg: lc_history.add_ai_message(ai_msg)

        # 3. Invoke the chain
        config = {"configurable": {"session_id": session_id}}
        response_message = chain.invoke({"input": user_input}, config=config)
        ai_response = response_message.content if hasattr(response_message, 'content') else str(response_message)

        # 4. Update UI history list
        chat_history_list.append((user_input, ai_response))
        yield "", chat_history_list # Clear input, return updated history

    except Exception as e:
        error_msg = f"🤖 Oops! An error occurred: {e}"
        print(f"Chatbot Error: {error_msg}") # Log for debugging
        chat_history_list.append((user_input, error_msg))
        yield "", chat_history_list

# --- Pomodoro Timer Logic ---
def update_timer(current_time_sec, timer_running, current_mode, pomodoro_count):
    """Updates timer display and state every second."""
    should_stop_timer = False
    if timer_running and current_time_sec > 0:
        current_time_sec -= 1
    elif timer_running and current_time_sec == 0:
        should_stop_timer = True
        if current_mode == "Work":
            current_mode = "Break"
            current_time_sec = POMODORO_BREAK_MINS * 60
            pomodoro_count += 1
        else:
            current_mode = "Work"
            current_time_sec = POMODORO_WORK_MINS * 60

    mins, secs = divmod(current_time_sec, 60)
    time_display = f"{mins:02d}:{secs:02d}"
    mode_display = f"Mode: {current_mode} Pomodoros: {pomodoro_count}"

    if should_stop_timer: timer_running = False
    start_btn_text = "Pause" if timer_running else "Start"
    start_btn_variant = "secondary" if timer_running else "primary"

    return (
        current_time_sec, timer_running, current_mode, pomodoro_count,
        time_display, mode_display,
        gr.update(value=start_btn_text, variant=start_btn_variant), gr.update(interactive=True)
    )

def start_pause_timer(current_time_sec, timer_running, current_mode):
    """Toggles timer running state."""
    new_running_state = not timer_running
    if new_running_state and current_time_sec == 0: # Reset if starting at 0
        current_time_sec = POMODORO_WORK_MINS * 60 if current_mode == "Work" else POMODORO_BREAK_MINS * 60
    return new_running_state, current_time_sec

def reset_timer():
    """Resets timer to initial work state."""
    return POMODORO_WORK_MINS * 60, False, "Work", 0

# --- API Key Handling ---
def validate_api_key(new_api_key):
    """Validates the API key by attempting LLM initialization."""
    session_id = "prod_pal_session"
    if session_id in chat_history_store:
        chat_history_store[session_id].clear() # Clear history on key change

    if not new_api_key:
        return None, [(None, "API Key is required!")], "API Key cleared."

    try:
        # Attempt to initialize LLM as validation
        ChatGoogleGenerativeAI(model=DEFAULT_MODEL, google_api_key=new_api_key, temperature=0.1)
        status_msg = "API Key validated. Ready!"
        initial_chat = [(None, "Hi there! I'm Productivity Pal. How can I help?")]
        return new_api_key, initial_chat, status_msg
    except Exception as e:
        error_msg = f"API Key Validation Error: {e}"
        print(error_msg) # Log for debugging
        initial_chat = [(None, f"⚠️ {error_msg}")]
        return new_api_key, initial_chat, "API Key Error."

# --- Build Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal", secondary_hue="cyan"), title=APP_TITLE) as demo:

    # --- State ---
    env_api_key = os.getenv("GOOGLE_API_KEY")
    initial_api_key, initial_history, initial_status = validate_api_key(env_api_key)

    api_key_state = gr.State(initial_api_key)
    chat_history_state = gr.State(initial_history) # For UI display list
    timer_running_state = gr.State(False)
    remaining_time_state = gr.State(POMODORO_WORK_MINS * 60)
    current_mode_state = gr.State("Work")
    pomodoro_count_state = gr.State(0)

    # --- UI Layout ---
    gr.Markdown(f"<h1 style='text-align: center;'>{APP_TITLE}</h1>")
    gr.Markdown("<p style='text-align: center;'>AI Partner for Task Management, Note-Taking, Motivation & Goal Setting and Brainstorming Productivity Ideas...</p>")

    with gr.Row():
        # Main Chat Area
        with gr.Column(scale=3):
            chatbot_display = gr.Chatbot(label="Chat", height=550, value=initial_history)
            text_input = gr.Textbox(label="Your message:", placeholder="Ask for task help, motivation...", lines=2)
            with gr.Row():
                submit_button = gr.Button("Send", variant="primary", scale=1)
                clear_button = gr.ClearButton([text_input, chat_history_state], value="Clear Chat", scale=1)

        # Sidebar Area
        with gr.Column(scale=1):
            # Settings
            with gr.Accordion("⚙️ Settings & Info", open=not initial_api_key): # Open if no key initially
                api_key_input = gr.Textbox(label="🔑 Google API Key", type="password", value=initial_api_key or "", interactive=True)
                api_status_display = gr.Markdown(f"Status: {initial_status}")
                gr.Markdown("---")
                gr.Markdown("**💡 Tips:** Break tasks down. Use time blocking. Try the Pomodoro timer below!")

            # Pomodoro Timer
            gr.Markdown("---")
            gr.Markdown("### 🍅 Pomodoro Timer")
            timer_display = gr.Label(f"{POMODORO_WORK_MINS:02d}:00", label="Time Left")
            gr.Markdown("---")
            mode_display = gr.Label("Mode: Work | Pomodoros: 0", label="Status")
            with gr.Row():
                start_pause_button = gr.Button("Start", variant="primary", size="sm")
                reset_button = gr.Button("Reset", size="sm")

    # --- Event Handlers ---

    # API Key Update
    def handle_api_key_update_wrapper(new_api_key):
        key, history, status = validate_api_key(new_api_key)
        return key, history, status, history # Update state & UI components

    api_key_input.submit(
        handle_api_key_update_wrapper,
        inputs=[api_key_input],
        outputs=[api_key_state, chat_history_state, api_status_display, chatbot_display]
    )

    # Chat Submission
    def handle_chat_submit_wrapper(api_key, user_input, history_list):
        """Wrapper because chatbot_response is a generator."""
        final_output, final_history = "", history_list
        for out, hist in chatbot_response(api_key, user_input, history_list):
            final_output, final_history = out, hist
        return final_output, final_history, final_history # Update input, state, UI

    submit_triggers = [text_input.submit, submit_button.click]
    for trigger in submit_triggers:
        trigger(
            handle_chat_submit_wrapper,
            inputs=[api_key_state, text_input, chat_history_state],
            outputs=[text_input, chat_history_state, chatbot_display],
            api_name=False
        )

    # Clear Chat Button
    def clear_chat_action_wrapper():
        session_id = "prod_pal_session"
        if session_id in chat_history_store:
            chat_history_store[session_id].clear() # Clear LangChain history too
        initial_message = [(None, "Chat cleared.")]
        return "", initial_message, initial_message # Clear input, set UI history state & display

    clear_button.click(
        clear_chat_action_wrapper,
        inputs=[],
        outputs=[text_input, chat_history_state, chatbot_display]
    )

    # Pomodoro Timer Buttons
    start_pause_button.click(
        start_pause_timer,
        inputs=[remaining_time_state, timer_running_state, current_mode_state],
        outputs=[timer_running_state, remaining_time_state],
        api_name=False
    )
    reset_button.click(
        reset_timer,
        inputs=[],
        outputs=[remaining_time_state, timer_running_state, current_mode_state, pomodoro_count_state]
    ).then( # Chain UI update immediately after reset logic
        update_timer,
        inputs=[remaining_time_state, timer_running_state, current_mode_state, pomodoro_count_state],
        outputs=[
            remaining_time_state, timer_running_state, current_mode_state, pomodoro_count_state,
            timer_display, mode_display, start_pause_button, reset_button
        ],
        api_name=False
    )

    # Pomodoro Timer Update Loop
    demo.load(
        update_timer,
        inputs=[remaining_time_state, timer_running_state, current_mode_state, pomodoro_count_state],
        outputs=[
            remaining_time_state, timer_running_state, current_mode_state, pomodoro_count_state,
            timer_display, mode_display, start_pause_button, reset_button
        ],
        every=1,
        api_name=False
    )

# --- Run the App ---
if __name__ == "__main__":
    demo.launch(debug=False, share=False) # share=True for public link