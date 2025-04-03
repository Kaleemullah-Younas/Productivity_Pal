import gradio as gr
import os
import time
from dotenv import load_dotenv

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- Configuration ---
load_dotenv()

# --- Constants ---
APP_TITLE = "Productivity Pal ✨"
DEFAULT_MODEL = "gemini-1.5-flash-latest"
MAX_HISTORY_PAIRS = 5 # Keep last 5 user/AI message pairs
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
    session_id = "prod_pal_session" 

    if not api_key:
        yield "", chat_history_list + [(user_input, "⚠️ Error: API Key is missing.")]
        return
    if not user_input or not user_input.strip(): 
         yield "", chat_history_list
         return

    try:
        chain = create_llm_chain(api_key)

        config = {"configurable": {"session_id": session_id}}
        response_message = chain.invoke({"input": user_input}, config=config)
        ai_response = response_message.content if hasattr(response_message, 'content') else str(response_message)

        # 4. Update UI history list (Gradio's format)
        chat_history_list.append((user_input, ai_response))

        # Optional: Prune Gradio's UI history list if it exceeds the desired display length
        if len(chat_history_list) > MAX_HISTORY_PAIRS + 5: 
             chat_history_list = chat_history_list[-(MAX_HISTORY_PAIRS + 5):]


        yield "", chat_history_list 

    except Exception as e:
        error_msg = f"🤖 Oops! An error occurred: {e}"
        print(f"Chatbot Error: {e}") 
        import traceback
        traceback.print_exc()
        chat_history_list.append((user_input if user_input else "[Input Error]", error_msg))
        yield "", chat_history_list


# --- Pomodoro Timer Logic ---
def update_timer(current_time_sec, timer_running, current_mode, pomodoro_count):
    """Updates timer display and state every second."""
    original_timer_running = timer_running 
    should_stop_timer = False

    if timer_running and current_time_sec > 0:
        current_time_sec -= 1
    elif timer_running and current_time_sec == 0:
        # Timer finished, switch mode
        if current_mode == "Work":
            current_mode = "Break"
            current_time_sec = POMODORO_BREAK_MINS * 60
            pomodoro_count += 1
        else: 
            current_mode = "Work"
            current_time_sec = POMODORO_WORK_MINS * 60
        timer_running = False
        should_stop_timer = True 

    mins, secs = divmod(current_time_sec, 60)
    time_display = f"{mins:02d}:{secs:02d}"
    mode_display = f"Mode: {current_mode} | Pomodoros: {pomodoro_count}"

    start_btn_text = "Pause" if timer_running else "Start"
    start_btn_variant = "secondary" if timer_running else "primary"
    reset_interactive = not timer_running

    return (
        current_time_sec, timer_running, current_mode, pomodoro_count,
        time_display, mode_display,
        gr.update(value=start_btn_text, variant=start_btn_variant), 
        gr.update(interactive=reset_interactive) 
    )

def start_pause_timer(current_time_sec, timer_running, current_mode):
    """Toggles timer running state."""
    new_running_state = not timer_running
    # If starting from 0, reset to the current mode's duration
    if new_running_state and current_time_sec == 0:
        current_time_sec = POMODORO_WORK_MINS * 60 if current_mode == "Work" else POMODORO_BREAK_MINS * 60
        
    if not new_running_state and current_time_sec == 0:
         pass 

    return new_running_state, current_time_sec

def reset_timer():
    """Resets timer to initial work state."""
    # Return values to update the states directly
    return POMODORO_WORK_MINS * 60, False, "Work", 0

# --- API Key Handling ---
def validate_api_key(new_api_key):
    """Validates the API key by attempting LLM initialization. Returns status message and initial history."""
    session_id = "prod_pal_session"
    if session_id in chat_history_store:
        chat_history_store[session_id].clear()

    if not new_api_key:
        return None, [(None, "API Key is required!")], "API Key cleared. Please enter a valid key."

    try:
        ChatGoogleGenerativeAI(model=DEFAULT_MODEL, google_api_key=new_api_key, temperature=0.1).invoke("test")
        status_msg = "✅ API Key validated. Ready!"
        initial_chat = [(None, "Hi there! I'm Productivity Pal. How can I help you be more productive today?")]
        return new_api_key, initial_chat, status_msg
    except Exception as e:
        error_msg = f"API Key Validation Error: {str(e)}"
        print(f"API Key Error Details: {e}") # Log full error for debugging
        initial_chat = [(None, f"⚠️ Invalid API Key or Connection Error. Please check your key and network. ({e})")]
        return new_api_key, initial_chat, f"❌ API Key Error: {e}"


# --- Build Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal", secondary_hue="cyan"), title=APP_TITLE) as demo:

    # --- State ---
    # Try loading API key from environment
    env_api_key = os.getenv("GOOGLE_API_KEY")
    # Perform initial validation
    initial_api_key_value, initial_history_value, initial_status_msg = validate_api_key(env_api_key)

    # Initialize state components
    api_key_state = gr.State(initial_api_key_value)
    # chat_history_state stores the Gradio format list: [(user, ai), (user, ai), ...]
    chat_history_state = gr.State(initial_history_value)
    timer_running_state = gr.State(False)
    remaining_time_state = gr.State(POMODORO_WORK_MINS * 60)
    current_mode_state = gr.State("Work")
    pomodoro_count_state = gr.State(0)

    # --- UI Layout ---
    gr.Markdown(f"<h1 style='text-align: center;'>{APP_TITLE}</h1>")
    gr.Markdown("<p style='text-align: center;'>Your AI Partner for Task Management, Note-Taking, Motivation & Productivity Ideas.</p>")

    with gr.Row():
        # Main Chat Area
        with gr.Column(scale=3):
            # Initialize chatbot display with the initial history from validation
            chatbot_display = gr.Chatbot(
                label="Chat",
                height=550,
                value=initial_history_value, 
                bubble_full_width=False
            )
            text_input = gr.Textbox(
                label="Your message:",
                placeholder="Ask for task help, motivation, pomodoro tips...",
                lines=2,
                interactive=bool(initial_api_key_value)
            )
            with gr.Row():
                submit_button = gr.Button(
                    "Send",
                    variant="primary",
                    scale=1,
                    interactive=bool(initial_api_key_value)
                )
                clear_button = gr.ClearButton(
                    [text_input, chat_history_state],
                    value="Clear Chat",
                    scale=1,
                )

        # Sidebar Area
        with gr.Column(scale=1):
            # Settings
            with gr.Accordion("⚙️ Settings & Info", open=not bool(initial_api_key_value)):
                api_key_input = gr.Textbox(
                    label="🔑 Google API Key",
                    type="password",
                    # Use the value from validation (might be None or the env var)
                    value=env_api_key or "",
                    interactive=True,
                    placeholder="Enter your Google Gemini API Key"
                 )
                # Use Markdown for better formatting, initialize with validation status
                api_status_display = gr.Markdown(f"Status: {initial_status_msg}")
                gr.Markdown("---")
                gr.Markdown("**💡 Tips:** Break tasks down. Use time blocking. Try the Pomodoro timer below!")

            # Pomodoro Timer
            gr.Markdown("---")
            gr.Markdown("### 🍅 Pomodoro Timer")
            timer_display = gr.Label(f"{POMODORO_WORK_MINS:02d}:00", label="Time Left")
            mode_display = gr.Label("Mode: Work Pomodoros: 0", label="Status")
            with gr.Row():
                start_pause_button = gr.Button("Start", variant="primary", size="sm")
                # Make reset initially interactive
                reset_button = gr.Button("Reset", size="sm", interactive=True)

    # --- Event Handlers ---

    # API Key Update / Validation
    def handle_api_key_update_wrapper(new_api_key):
        """Wrapper to update state and UI components after validation."""
        key_state_val, history_val, status_msg = validate_api_key(new_api_key)
        # Enable/disable chat based on validation result
        chat_interactive = bool(key_state_val)
        return (
            key_state_val,         
            history_val,           
            f"Status: {status_msg}", 
            history_val,          
            gr.update(interactive=chat_interactive), 
            gr.update(interactive=chat_interactive)  
        )

    api_key_input.submit(
        handle_api_key_update_wrapper,
        inputs=[api_key_input],
        outputs=[
            api_key_state,
            chat_history_state,
            api_status_display,
            chatbot_display,
            text_input,         
            submit_button        
        ],
        api_name=False 
    )

    # Chat Submission
    def handle_chat_submit_wrapper(api_key, user_input, history_list):
        """Wrapper because chatbot_response is a generator."""
        final_output, final_history = "", history_list
        if not user_input or not user_input.strip():
             return "", history_list, history_list 

        history_list_with_user = history_list + [(user_input, None)]

        # Use a try-except block here as well for safety during generation
        try:
            # Pass the latest history list (without the user message just added to UI)
            # The chatbot_response function will handle adding both user and AI messages
            generator = chatbot_response(api_key, user_input, history_list)
            final_output, final_history = next(generator)

        except Exception as e:
            print(f"Error during chatbot response generation: {e}")
            error_msg = f"🤖 Apologies, I encountered an error processing that request: {e}"
            final_history = history_list + [(user_input, error_msg)]
            final_output = "" 

        return final_output, final_history, final_history

    submit_triggers = [text_input.submit, submit_button.click]
    for trigger in submit_triggers:
        trigger(
            handle_chat_submit_wrapper,
            inputs=[api_key_state, text_input, chat_history_state],
            outputs=[text_input, chat_history_state, chatbot_display],
            api_name=False
        )

    # Clear Chat Button Action
    def clear_chat_action_wrapper():
        """Clears LangChain history and resets Gradio chat UI."""
        session_id = "prod_pal_session"
        if session_id in chat_history_store:
            chat_history_store[session_id].clear() 
        initial_message = [(None, "Chat cleared. Ready for a fresh start!")]
        return "", initial_message, initial_message 

    # Connect the wrapper function to the clear button's click event
    clear_button.click(
        clear_chat_action_wrapper,
        inputs=[],
        outputs=[text_input, chat_history_state, chatbot_display],
        api_name=False
    )


    # --- Pomodoro Timer Event Handlers ---

    # Start/Pause Button Click
    start_pause_button.click(
        start_pause_timer,
        inputs=[remaining_time_state, timer_running_state, current_mode_state],
        outputs=[timer_running_state, remaining_time_state],
        api_name=False
    ).then( 
        update_timer,
         inputs=[remaining_time_state, timer_running_state, current_mode_state, pomodoro_count_state],
         outputs=[ 
             remaining_time_state, timer_running_state, current_mode_state, pomodoro_count_state,
             timer_display, mode_display, start_pause_button, reset_button
         ],
         api_name=False
    )

    # Reset Button Click
    reset_button.click(
        reset_timer,
        inputs=[],
        outputs=[remaining_time_state, timer_running_state, current_mode_state, pomodoro_count_state],
        api_name=False
    ).then( 
        update_timer, 
        inputs=[remaining_time_state, timer_running_state, current_mode_state, pomodoro_count_state],
        outputs=[ 
            remaining_time_state, timer_running_state, current_mode_state, pomodoro_count_state,
            timer_display, mode_display, start_pause_button, reset_button
        ],
        api_name=False
    )

    # Pomodoro Timer
    demo.load(
        update_timer,
        inputs=[remaining_time_state, timer_running_state, current_mode_state, pomodoro_count_state],
        outputs=[ 
            remaining_time_state, 
            timer_running_state,  
            current_mode_state,   
            pomodoro_count_state, 
            timer_display,        
            mode_display,         
            start_pause_button,   
            reset_button          
        ],
        every=1, # Run every 1 second
        api_name=False
    )

# --- Run the App ---
if __name__ == "__main__":
    demo.launch(debug=False, share=True)
