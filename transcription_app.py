import streamlit as st
import anthropic
import os
import json
import time
from PIL import Image
import io
import base64
from datetime import datetime
import mimetypes

# Import Gemini SDK
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    st.warning("Google Gemini SDK inte installerad. Kör: pip install google-genai")

# Set page configuration
st.set_page_config(
    page_title="Transkriptionsassistent för Manuskript",
    page_icon="📜",
    layout="wide"
)

# Initialize Anthropic client
@st.cache_resource
def get_anthropic_client(_api_key):
    """Initialize Anthropic client with provided API key"""
    return anthropic.Anthropic(api_key=_api_key)

# Initialize Gemini client
@st.cache_resource
def get_gemini_client(_api_key):
    """Initialize Gemini client with provided API key"""
    if not GEMINI_AVAILABLE:
        raise ValueError("Google Gemini SDK inte installerad.")
    return genai.Client(api_key=_api_key)

# Convert image to base64 for Anthropic API
def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Convert PIL image to bytes for Gemini API
def image_to_bytes(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return buffered.getvalue()

# Initialize session state variables
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "current_workflow_stage" not in st.session_state:
    st.session_state.current_workflow_stage = "upload"
if "current_iteration" not in st.session_state:
    st.session_state.current_iteration = 0
if "default_prompt" not in st.session_state:
    st.session_state.default_prompt = "Transkribera den handskrivna texten i bilden så noggrant som möjligt. Läs rad för rad, ord för ord. Returnera endast transkriptionen, inget annat!"
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "training"  # "training" or "direct"
if "direct_mode_type" not in st.session_state:
    st.session_state.direct_mode_type = "Enstaka sida"  # "Enstaka sida" or "Bulk-transkription (flera sidor)"
if "training_metadata" not in st.session_state:
    st.session_state.training_metadata = {
        "name": "Onamngiven träningssession",
        "description": "Ingen beskrivning",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "iterations": 0
    }
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Claude"  # "Claude" or "Gemini"
if "anthropic_api_key" not in st.session_state:
    st.session_state.anthropic_api_key = ""
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = ""

# Function to save training history to JSON
def save_training_history():
    data = {
        "conversation_history": st.session_state.conversation_history,
        "metadata": st.session_state.training_metadata
    }
    return json.dumps(data, ensure_ascii=False)

# Function to load training history from JSON
def load_training_history(json_string):
    try:
        data = json.loads(json_string)
        st.session_state.conversation_history = data["conversation_history"]
        st.session_state.training_metadata = data["metadata"]
        return True
    except Exception as e:
        st.error(f"Fel vid inläsning av träningshistorik: {str(e)}")
        return False

# Function to process transcription with Claude
def process_transcription_claude(image, prompt, update_history=True):
    if not st.session_state.anthropic_api_key:
        raise ValueError("Ingen Anthropic API-nyckel angiven. Vänligen ange din API-nyckel i sidofältet.")
    
    client = get_anthropic_client(st.session_state.anthropic_api_key)
    base64_image = image_to_base64(image)
    
    # Construct the complete message history for context
    messages = []
    
    # Add all previous conversation history
    for msg in st.session_state.conversation_history:
        messages.append(msg)
    
    # Create the user message with the current image and prompt
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64_image
                }
            },
            {
                "type": "text",
                "text": prompt
            }
        ]
    }
    
    # Add to the messages for the API call
    messages.append(user_message)
    
    # Call Claude API
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=4000,
        messages=messages
    )
    
    transcription = response.content[0].text
    
    # If we should update the history (training mode), add the exchange to conversation history
    if update_history:
        st.session_state.conversation_history.append(user_message)
        st.session_state.conversation_history.append({
            "role": "assistant",
            "content": transcription
        })
    
    return transcription

# Function to process transcription with Gemini
def process_transcription_gemini(image, prompt, update_history=True):
    if not st.session_state.gemini_api_key:
        raise ValueError("Ingen Google AI API-nyckel angiven. Vänligen ange din API-nyckel i sidofältet.")
    
    client = get_gemini_client(st.session_state.gemini_api_key)
    image_bytes = image_to_bytes(image)
    mime_type = "image/png"
    
    # Build content - simplified approach for first message
    if len(st.session_state.conversation_history) == 0:
        # Simple first message without history
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        inline_data=types.Blob(
                            mime_type=mime_type,
                            data=image_bytes
                        )
                    ),
                    types.Part.from_text(text=prompt)
                ]
            )
        ]
    else:
        # Build content from conversation history
        contents = []
        
        # Add all previous conversation history
        for msg in st.session_state.conversation_history:
            role = msg["role"]
            if role == "assistant":
                role = "model"  # Gemini uses "model" instead of "assistant"
            
            # Handle message content
            if isinstance(msg["content"], str):
                # Simple text message
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=msg["content"])]
                    )
                )
            elif isinstance(msg["content"], list):
                # Complex message with image and text
                parts = []
                for item in msg["content"]:
                    if item["type"] == "text":
                        parts.append(types.Part.from_text(text=item["text"]))
                    elif item["type"] == "image":
                        # Extract base64 data and convert back to bytes
                        img_data = base64.b64decode(item["source"]["data"])
                        parts.append(
                            types.Part(
                                inline_data=types.Blob(
                                    mime_type="image/png",
                                    data=img_data
                                )
                            )
                        )
                contents.append(types.Content(role=role, parts=parts))
        
        # Add current user message
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        inline_data=types.Blob(
                            mime_type=mime_type,
                            data=image_bytes
                        )
                    ),
                    types.Part.from_text(text=prompt)
                ]
            )
        )
    
    # Call Gemini API
    try:
        # Use explicit model path
        model_name = "models/gemini-2.5-pro"
        
        # Configure the generation
        generate_content_config = types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=8000,  # Ökat från 4000 för längre transkriptioner
        )
        
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=generate_content_config,
        )
        
        # Extract text from response - simplified and more robust
        transcription = None
        
        # Primary method: use response.text
        try:
            transcription = response.text
            if transcription:  # Make sure it's not None or empty
                st.success("✅ Gemini svarade med response.text")
            else:
                transcription = None  # Reset if empty
        except Exception as e:
            # response.text threw an exception, try alternative
            st.warning(f"⚠️ Kunde inte använda response.text: {e}")
            transcription = None
        
        # Fallback method: extract from candidates
        if not transcription:
            try:
                if hasattr(response, 'candidates') and response.candidates:
                    if response.candidates[0].content and response.candidates[0].content.parts:
                        transcription = response.candidates[0].content.parts[0].text
                        if transcription:
                            st.success("✅ Gemini svarade med candidates[0]")
            except Exception as e:
                st.warning(f"⚠️ Kunde inte extrahera från candidates: {e}")
        
        # Check finish reason for specific issues
        if not transcription and hasattr(response, 'candidates') and response.candidates:
            finish_reason = response.candidates[0].finish_reason
            
            # Even if MAX_TOKENS, there might be partial text we can use
            if str(finish_reason) == "MAX_TOKENS" or "MAX_TOKENS" in str(finish_reason):
                # Try to get whatever partial text exists
                try:
                    if response.candidates[0].content and response.candidates[0].content.parts:
                        partial_text = response.candidates[0].content.parts[0].text
                        if partial_text:
                            st.warning(
                                "⚠️ Gemini nådde max token-limit! "
                                "Visar partiell transkription (kan vara ofullständig)."
                            )
                            return partial_text
                except:
                    pass
                
                # If we couldn't get partial text, raise error
                raise ValueError(
                    "⚠️ Gemini nådde max token-limit!\n\n"
                    "Transkriptionen är för lång för den nuvarande inställningen.\n"
                    "Detta kan bero på:\n"
                    "1. Bilden innehåller mycket text\n"
                    "2. Gemini genererade en väldigt detaljerad beskrivning\n\n"
                    "Försök med:\n"
                    "- En mer kortfattad prompt (t.ex. 'Transkribera endast texten, inget annat.')\n"
                    "- Dela upp bilden i mindre delar\n"
                    "- Använd Claude som kan hantera längre svar"
                )
        
        if not transcription:
            # Debug information
            error_msg = "Gemini returnerade ett tomt svar.\n\n"
            error_msg += f"Response typ: {type(response)}\n"
            if hasattr(response, 'candidates'):
                error_msg += f"Antal candidates: {len(response.candidates)}\n"
                if response.candidates:
                    error_msg += f"Finish reason: {response.candidates[0].finish_reason}\n"
                    if hasattr(response.candidates[0], 'safety_ratings'):
                        error_msg += f"Safety ratings: {response.candidates[0].safety_ratings}\n"
            error_msg += "\nKontrollera att:\n"
            error_msg += "1. Din API-nyckel är giltig\n"
            error_msg += "2. Bilden är läsbar\n"
            error_msg += "3. Du har tillgång till Gemini 2.5 Pro\n"
            raise ValueError(error_msg)
        
        # If we should update the history (training mode), add the exchange to conversation history
        if update_history:
            # Store in Claude's format for compatibility
            user_message = {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64.b64encode(image_bytes).decode("utf-8")
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
            st.session_state.conversation_history.append(user_message)
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": transcription
            })
        
        return transcription
        
    except Exception as e:
        raise Exception(f"Gemini API fel: {str(e)}")

# Main function to handle the transcription process
def process_transcription(image, prompt, update_history=True):
    if st.session_state.selected_model == "Claude":
        return process_transcription_claude(image, prompt, update_history)
    elif st.session_state.selected_model == "Gemini":
        return process_transcription_gemini(image, prompt, update_history)
    else:
        raise ValueError(f"Okänd modell: {st.session_state.selected_model}")

# Main app title
st.title("Transkriptionsassistent")
st.write("Ladda upp bilder av handskrivna manuskript och träna AI att transkribera dem korrekt.")

# App mode selector
st.sidebar.header("Appläge")
app_mode = st.sidebar.radio(
    "Välj läge:",
    ["Träningsläge", "Direktläge"],
    index=0 if st.session_state.app_mode == "training" else 1
)

# Update app mode in session state
st.session_state.app_mode = "training" if app_mode == "Träningsläge" else "direct"

# Sidebar for app controls and settings
with st.sidebar:
    st.header("Inställningar")
    
    # Model selector
    st.subheader("Välj AI-modell")
    
    model_options = ["Claude"]
    if GEMINI_AVAILABLE:
        model_options.append("Gemini")
    
    selected_model = st.selectbox(
        "AI-modell:",
        model_options,
        index=model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0
    )
    
    # Update selected model in session state
    st.session_state.selected_model = selected_model
    
    # API Key input based on selected model
    st.divider()
    st.subheader("API-nyckel")
    
    if selected_model == "Claude":
        st.info("🤖 **Claude Sonnet 4.5**")
        api_key_input = st.text_input(
            "Ange din Anthropic API-nyckel:",
            type="password",
            value=st.session_state.anthropic_api_key,
            help="Din API-nyckel från https://console.anthropic.com/"
        )
        st.session_state.anthropic_api_key = api_key_input
        
        if not st.session_state.anthropic_api_key:
            st.warning("⚠️ Vänligen ange din Anthropic API-nyckel för att använda Claude.")
        else:
            st.success("✅ API-nyckel angiven")
            
    elif selected_model == "Gemini":
        st.info("✨ **Gemini 2.5 Pro**")
        api_key_input = st.text_input(
            "Ange din Google AI API-nyckel:",
            type="password",
            value=st.session_state.gemini_api_key,
            help="Din API-nyckel från https://aistudio.google.com/apikey"
        )
        st.session_state.gemini_api_key = api_key_input.strip()
        
        if not st.session_state.gemini_api_key:
            st.warning("⚠️ Vänligen ange din Google AI API-nyckel för att använda Gemini.")
        else:
            st.success("✅ API-nyckel angiven")
            
        # Tips för Gemini-användare
        with st.expander("💡 Tips för bästa resultat med Gemini"):
            st.markdown("""
            **För längre transkriptioner:**
            - Använd en kortare, tydligare prompt
            - Exempel: "Transkribera endast texten. Inga kommentarer."
            - För mycket långa texter, överväg att använda Claude istället
            
            **Max output-tokens:** 8000 (räcker för de flesta manuskript)
            """)

    # Editable default prompt
    st.divider()
    st.subheader("📝 Standard-prompt")
    
    edited_prompt = st.text_area(
        "Redigera standardprompten för transkription:",
        value=st.session_state.default_prompt,
        height=150,
        help="Denna prompt används för alla transkriptioner. Modifiera för att ge kontext till bilden som ska transkriberas"
    )
    
    # Update if changed
    if edited_prompt != st.session_state.default_prompt:
        st.session_state.default_prompt = edited_prompt
        st.success("✅ Prompt uppdaterad!")
    
    # Reset to default button
    col1, col2 = st.columns(2)
    with col1:
        if st.button("↺ Återställ till standard", use_container_width=True):
            st.session_state.default_prompt = "Transkribera den handskrivna texten i bilden så noggrant som möjligt. Läs rad för rad, ord för ord. Returnera endast transkriptionen, inget annat!"
            st.success("✅ Återställd!")
            st.rerun()
    with col2:
        # Show character count
        st.metric("Tecken", len(st.session_state.default_prompt))
    
    # Training history management
    st.divider()
    
    # In direct mode, allow loading training history
    if st.session_state.app_mode == "direct":
        st.header("Ladda träningshistorik")
        uploaded_history = st.file_uploader("Välj en sparad träningsfil (.json)", type=["json"])
        
        if uploaded_history is not None:
            if st.button("Ladda träningshistorik"):
                content = uploaded_history.read().decode("utf-8")
                success = load_training_history(content)
                if success:
                    st.success(f"Träningshistorik laddad: {st.session_state.training_metadata['name']} ({st.session_state.training_metadata['iterations']} iterationer)")
                    # Reset workflow stage but keep conversation history
                    st.session_state.current_workflow_stage = "upload"
                    if "current_image" in st.session_state:
                        del st.session_state.current_image
                    if "current_transcription" in st.session_state:
                        del st.session_state.current_transcription
                    st.rerun()
    
    # In training mode, allow saving training history
    elif st.session_state.app_mode == "training" and len(st.session_state.conversation_history) > 0:
        st.header("Spara träningshistorik")
        
        # Edit metadata for the training session
        st.session_state.training_metadata["name"] = st.text_input(
            "Namn på träningssessionen:", 
            value=st.session_state.training_metadata["name"]
        )
        
        st.session_state.training_metadata["description"] = st.text_area(
            "Beskrivning:", 
            value=st.session_state.training_metadata["description"],
            height=100
        )
        
        # Update metadata
        st.session_state.training_metadata["iterations"] = st.session_state.current_iteration
        st.session_state.training_metadata["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if st.button("Spara träningshistorik"):
            json_data = save_training_history()
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_history_{now}.json"
            
            st.download_button(
                label="Ladda ner träningshistorik",
                data=json_data,
                file_name=filename,
                mime="application/json"
            )
        
        # Option to reset
        if st.button("Återställ träningshistorik"):
            if st.button("Bekräfta återställning"):
                st.session_state.conversation_history = []
                st.session_state.current_iteration = 0
                st.session_state.current_workflow_stage = "upload"
                st.session_state.training_metadata = {
                    "name": "Onamngiven träningssession",
                    "description": "Ingen beskrivning",
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "iterations": 0
                }
                # Clear current image and transcription if they exist
                if "current_image" in st.session_state:
                    del st.session_state.current_image
                if "current_transcription" in st.session_state:
                    del st.session_state.current_transcription
                st.rerun()
    
    # Universal reset button at the bottom
    st.divider()
    st.subheader("🔄 Rensa och börja om")
    
    # Initialize confirmation state
    if "confirm_reset" not in st.session_state:
        st.session_state.confirm_reset = False
    
    if not st.session_state.confirm_reset:
        if st.button("🗑️ Rensa allt och börja om", type="primary", use_container_width=True):
            st.session_state.confirm_reset = True
            st.rerun()
    else:
        st.warning("⚠️ Är du säker? Detta raderar all träningsdata och aktuella transkriptioner!")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Ja, rensa allt", use_container_width=True):
                # Clear all session state
                st.session_state.conversation_history = []
                st.session_state.current_iteration = 0
                st.session_state.current_workflow_stage = "upload"
                st.session_state.training_metadata = {
                    "name": "Onamngiven träningssession",
                    "description": "Ingen beskrivning",
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "iterations": 0
                }
                
                # Clear training mode data
                if "current_image" in st.session_state:
                    del st.session_state.current_image
                if "current_transcription" in st.session_state:
                    del st.session_state.current_transcription
                
                # Clear direct mode data
                if "direct_mode_image" in st.session_state:
                    del st.session_state.direct_mode_image
                if "direct_transcription" in st.session_state:
                    del st.session_state.direct_transcription
                
                # Clear bulk mode data
                if "bulk_transcription_results" in st.session_state:
                    st.session_state.bulk_transcription_results = []
                if "bulk_transcription_completed" in st.session_state:
                    st.session_state.bulk_transcription_completed = False
                
                st.session_state.confirm_reset = False
                st.success("✅ Allt har rensats!")
                time.sleep(1)
                st.rerun()
        with col2:
            if st.button("❌ Avbryt", use_container_width=True):
                st.session_state.confirm_reset = False
                st.rerun()

# Main content area
if st.session_state.app_mode == "training":
    # TRAINING MODE
    col1, col2 = st.columns([4, 1])
    with col1:
        st.header("Träningsläge")
    with col2:
        if st.button("🔄 Börja om", help="Avbryt och börja om från början"):
            st.session_state.current_workflow_stage = "upload"
            if "current_image" in st.session_state:
                del st.session_state.current_image
            if "current_transcription" in st.session_state:
                del st.session_state.current_transcription
            st.rerun()
    
    st.write("I detta läge kan du träna AI genom att ge feedback på transkriptioner.")
    
    # Show current iteration
    if st.session_state.current_iteration > 0:
        st.info(f"Aktuell träningsiteration: {st.session_state.current_iteration}")
    
    # Step 1: Upload image
    if st.session_state.current_workflow_stage == "upload":
        st.subheader("Steg 1: Ladda upp en manuskriptbild")
        uploaded_file = st.file_uploader("Välj en bild av ett handskrivet manuskript", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Load and display the image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uppladdad manuskriptbild", use_column_width=True)
            
            # Store in session state
            st.session_state.current_image = image
            
            # Move to next stage
            if st.button("Gå vidare till transkription"):
                st.session_state.current_workflow_stage = "transcribe"
                st.rerun()
    
    # Step 2: Get AI transcription
    elif st.session_state.current_workflow_stage == "transcribe":
        st.subheader("Steg 2: AI transkriberar manuskriptet")
        
        # Display the image
        st.image(st.session_state.current_image, caption="Manuskriptbild", use_column_width=True)
        
        # Show the prompt that will be used
        st.write("**Prompt som skickas till AI:**")
        st.code(st.session_state.default_prompt)
        
        # Button to start transcription
        if st.button(f"Låt {st.session_state.selected_model} transkribera"):
            with st.spinner(f"{st.session_state.selected_model} transkriberar manuskriptet..."):
                try:
                    # Get transcription from AI using all previous training
                    transcription = process_transcription(
                        st.session_state.current_image, 
                        st.session_state.default_prompt,
                        update_history=False  # Don't add to history yet
                    )
                    
                    # Store the transcription
                    st.session_state.current_transcription = transcription
                    
                    # Increment iteration counter
                    st.session_state.current_iteration += 1
                    
                    # Move to feedback stage
                    st.session_state.current_workflow_stage = "feedback"
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Ett fel uppstod vid transkriberingen: {str(e)}")
    
    # Step 3: Provide feedback
    elif st.session_state.current_workflow_stage == "feedback":
        st.subheader("Steg 3: Granska och ge feedback")
        
        # Display the image again
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Manuskriptbild:**")
            st.image(st.session_state.current_image, use_column_width=True)
        
        with col2:
            st.write(f"**{st.session_state.selected_model}s transkription:**")
            st.text_area(
                "Transkription:", 
                value=st.session_state.current_transcription, 
                height=300,
                disabled=True
            )
        
        # Provide the correct transcription
        st.write("**Ge den korrekta transkriptionen:**")
        correct_transcription = st.text_area(
            "Skriv eller klistra in den korrekta transkriptionen här:",
            height=200,
            key="correct_transcription_input"
        )
        
        # Button to submit feedback
        if st.button("Skicka feedback och fortsätt träningen"):
            if correct_transcription:
                with st.spinner(f"{st.session_state.selected_model} reflekterar över feedbacken..."):
                    try:
                        # First, add the original transcription request to history
                        base64_image = image_to_base64(st.session_state.current_image)
                        user_message = {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": base64_image
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": st.session_state.default_prompt
                                }
                            ]
                        }
                        st.session_state.conversation_history.append(user_message)
                        st.session_state.conversation_history.append({
                            "role": "assistant",
                            "content": st.session_state.current_transcription
                        })
                        
                        # Now send the feedback
                        feedback_prompt = f"Här är den korrekta transkriptionen:\n\n{correct_transcription}\n\nReflektera över skillnaderna mellan din transkription och den korrekta transkriptionen. Vad kan du lära dig för att förbättra framtida transkriptioner?"
                        
                        st.session_state.conversation_history.append({
                            "role": "user",
                            "content": feedback_prompt
                        })
                        
                        # Get AI's reflection using the appropriate client
                        if st.session_state.selected_model == "Claude":
                            if not st.session_state.anthropic_api_key:
                                raise ValueError("Ingen Anthropic API-nyckel angiven.")
                            client = get_anthropic_client(st.session_state.anthropic_api_key)
                            response = client.messages.create(
                                model="claude-sonnet-4-5-20250929",
                                max_tokens=2000,
                                messages=st.session_state.conversation_history
                            )
                            reflection = response.content[0].text
                        elif st.session_state.selected_model == "Gemini":
                            if not st.session_state.gemini_api_key:
                                raise ValueError("Ingen Google AI API-nyckel angiven.")
                            client = get_gemini_client(st.session_state.gemini_api_key)
                            
                            # Convert history to Gemini format
                            contents = []
                            for msg in st.session_state.conversation_history:
                                role = msg["role"]
                                if role == "assistant":
                                    role = "model"
                                
                                if isinstance(msg["content"], str):
                                    contents.append(
                                        types.Content(
                                            role=role,
                                            parts=[types.Part.from_text(text=msg["content"])]
                                        )
                                    )
                                elif isinstance(msg["content"], list):
                                    parts = []
                                    for item in msg["content"]:
                                        if item["type"] == "text":
                                            parts.append(types.Part.from_text(text=item["text"]))
                                        elif item["type"] == "image":
                                            img_data = base64.b64decode(item["source"]["data"])
                                            parts.append(
                                                types.Part(
                                                    inline_data=types.Blob(
                                                        mime_type="image/png",
                                                        data=img_data
                                                    )
                                                )
                                            )
                                    contents.append(types.Content(role=role, parts=parts))
                            
                            generate_content_config = types.GenerateContentConfig(
                                temperature=0.7,
                                max_output_tokens=4000,  # Reflection behöver mindre tokens
                            )
                            
                            response = client.models.generate_content(
                                model="models/gemini-2.5-pro",
                                contents=contents,
                                config=generate_content_config,
                            )
                            
                            # Extract text from response - same robust method as transcription
                            reflection = None
                            
                            # Primary method: use response.text
                            try:
                                reflection = response.text
                                if reflection:
                                    st.success("✅ Gemini svarade med response.text")
                                else:
                                    reflection = None
                            except Exception as e:
                                st.warning(f"⚠️ Kunde inte använda response.text: {e}")
                                reflection = None
                            
                            # Fallback method: extract from candidates
                            if not reflection:
                                try:
                                    if hasattr(response, 'candidates') and response.candidates:
                                        if response.candidates[0].content and response.candidates[0].content.parts:
                                            reflection = response.candidates[0].content.parts[0].text
                                            if reflection:
                                                st.success("✅ Gemini svarade med candidates[0]")
                                except Exception as e:
                                    st.warning(f"⚠️ Kunde inte extrahera från candidates: {e}")
                            
                            if not reflection:
                                raise ValueError("Gemini returnerade ett tomt svar.")
                        
                        # Add reflection to history
                        st.session_state.conversation_history.append({
                            "role": "assistant",
                            "content": reflection
                        })
                        
                        # Show the reflection
                        st.success("Feedback skickad!")
                        st.write(f"**{st.session_state.selected_model}s reflektion:**")
                        st.write(reflection)
                        
                        # Reset for next iteration
                        st.session_state.current_workflow_stage = "upload"
                        if "current_image" in st.session_state:
                            del st.session_state.current_image
                        if "current_transcription" in st.session_state:
                            del st.session_state.current_transcription
                        
                        time.sleep(2)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Ett fel uppstod: {str(e)}")
            else:
                st.warning("Vänligen ange den korrekta transkriptionen.")

else:
    # DIRECT MODE
    col1, col2 = st.columns([4, 1])
    with col1:
        st.header("Direktläge")
    with col2:
        if st.button("🔄 Börja om", help="Rensa och börja om från början"):
            if "direct_mode_image" in st.session_state:
                del st.session_state.direct_mode_image
            if "direct_transcription" in st.session_state:
                del st.session_state.direct_transcription
            if "bulk_transcription_results" in st.session_state:
                st.session_state.bulk_transcription_results = []
            if "bulk_transcription_completed" in st.session_state:
                st.session_state.bulk_transcription_completed = False
            st.rerun()
    
    # Show loaded training info if available
    if len(st.session_state.conversation_history) > 0:
        st.success(f"✅ Träningshistorik laddad: {st.session_state.training_metadata['name']} ({st.session_state.training_metadata['iterations']} iterationer)")
    else:
        st.info("💡 Ingen träningshistorik laddad. AI kommer att använda grundinställningarna.")
    
    # Sub-mode selector for direct mode
    direct_mode_type = st.radio(
        "Välj direktlägestyp:",
        ["Enstaka sida", "Bulktranskription (flera sidor)"]
    )
    st.session_state.direct_mode_type = direct_mode_type
    
    if direct_mode_type == "Enstaka sida":
        # Single page transcription
        st.subheader("Direkttranskription av en sida")
        
        # File uploader for single image
        uploaded_file = st.file_uploader("Välj en bild av ett handskrivet manuskript", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Load and display the image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uppladdad manuskriptbild", use_column_width=True)
            
            # Store in session state
            st.session_state.direct_mode_image = image
            
            # Show the prompt input field
            direct_prompt = st.text_area(
                f"Prompt för {st.session_state.selected_model}:", 
                value=st.session_state.default_prompt,
                height=150
            )
            
            # Button to start direct transcription
            if st.button("Starta direkttranskription"):
                with st.spinner(f"{st.session_state.selected_model} transkriberar manuskriptet..."):
                    try:
                        # Get transcription from AI using all previous training
                        # Pass update_history=False to prevent adding to conversation history in direct mode
                        transcription = process_transcription(
                            st.session_state.direct_mode_image, 
                            direct_prompt,
                            update_history=False
                        )
                        
                        # Display the result
                        st.session_state.direct_transcription = transcription
                        
                    except Exception as e:
                        st.error(f"Ett fel uppstod vid transkriberingen: {str(e)}")
            
            # Show the direct transcription result if available
            if "direct_transcription" in st.session_state:
                st.subheader("Transkriptionsresultat")
                st.text_area(
                    f"{st.session_state.selected_model}s transkription:", 
                    value=st.session_state.direct_transcription, 
                    height=300
                )
                
                # Option to copy to clipboard
                if st.button("Kopiera till urklipp"):
                    st.code(st.session_state.direct_transcription)
                    st.success("Transkription kopierad till urklipp!")
                
                # Button to clear for next transcription
                if st.button("Rensa och transkribera en ny bild"):
                    if "direct_transcription" in st.session_state:
                        del st.session_state.direct_transcription
                    if "direct_mode_image" in st.session_state:
                        del st.session_state.direct_mode_image
                    st.rerun()
        else:
            st.info("Ladda upp en bild för att börja transkribera.")
    
    else:  # Bulk transcription mode
        st.subheader("Bulktranskription av flera manuskript")
        
        # Initialize session state for bulk transcription if not already done
        if "bulk_transcription_results" not in st.session_state:
            st.session_state.bulk_transcription_results = []
        
        if "bulk_transcription_completed" not in st.session_state:
            st.session_state.bulk_transcription_completed = False
        
        if "bulk_transcription_progress" not in st.session_state:
            st.session_state.bulk_transcription_progress = 0
        
        # Allow uploading multiple files
        uploaded_files = st.file_uploader(
            "Välj flera bilder av handskrivna manuskript", 
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )
        
        # Show the prompt input field
        bulk_prompt = st.text_area(
            f"Prompt för {st.session_state.selected_model} (används för alla bilder):", 
            value=st.session_state.default_prompt,
            height=150
        )
        
        # Button to start bulk transcription
        if uploaded_files and st.button("Starta bulktranskription"):
            # Reset results
            st.session_state.bulk_transcription_results = []
            st.session_state.bulk_transcription_completed = False
            st.session_state.bulk_transcription_progress = 0
            
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process each file
            for i, uploaded_file in enumerate(uploaded_files):
                # Update progress
                progress_percent = int(100 * i / len(uploaded_files))
                progress_bar.progress(progress_percent)
                status_text.text(f"Transkriberar fil {i+1} av {len(uploaded_files)}: {uploaded_file.name}")
                
                try:
                    # Process the image
                    image = Image.open(uploaded_file)
                    
                    # Get transcription from AI
                    transcription = process_transcription(
                        image, 
                        bulk_prompt,
                        update_history=False
                    )
                    
                    # Store the result
                    st.session_state.bulk_transcription_results.append({
                        "filename": uploaded_file.name,
                        "transcription": transcription
                    })
                    
                except Exception as e:
                    # Store the error
                    st.session_state.bulk_transcription_results.append({
                        "filename": uploaded_file.name,
                        "transcription": f"FEL: {str(e)}"
                    })
            
            # Complete the progress bar
            progress_bar.progress(100)
            status_text.text(f"Transkribering klar! {len(uploaded_files)} filer bearbetade.")
            
            # Mark as completed
            st.session_state.bulk_transcription_completed = True
            st.rerun()
        
        # If bulk transcription has been completed, show results and download option
        if st.session_state.bulk_transcription_completed and st.session_state.bulk_transcription_results:
            st.subheader("Bulk-transkriptionsresultat")
            
            # Create a DataFrame from the results
            import pandas as pd
            results_df = pd.DataFrame(st.session_state.bulk_transcription_results)
            
            # Display the results in a table
            st.dataframe(results_df)
            
            # Create CSV for download
            csv = results_df.to_csv(index=False)
            
            # Create a download button
            st.download_button(
                label="Ladda ner resultat som CSV",
                data=csv,
                file_name="transkriptionsresultat.csv",
                mime="text/csv"
            )
            
            # Option to clear results
            if st.button("Rensa resultat och transkribera nya filer"):
                st.session_state.bulk_transcription_results = []
                st.session_state.bulk_transcription_completed = False
                st.rerun()

# Show training history
with st.expander("Visa träningshistorik"):
    if len(st.session_state.conversation_history) > 0:
        iteration = 1
        message_index = 0
        
        while message_index < len(st.session_state.conversation_history):
            # Try to find a complete iteration (4 messages)
            if message_index + 3 < len(st.session_state.conversation_history):
                # Get the messages
                image_msg = st.session_state.conversation_history[message_index]
                transcription_msg = st.session_state.conversation_history[message_index + 1]
                
                # Check if this is a training iteration with feedback
                if message_index + 3 < len(st.session_state.conversation_history) and "Här är den korrekta transkriptionen" in st.session_state.conversation_history[message_index + 2]["content"]:
                    # This is a training iteration with feedback
                    feedback_msg = st.session_state.conversation_history[message_index + 2]
                    reflection_msg = st.session_state.conversation_history[message_index + 3]
                    
                    st.write(f"### Träningsiteration {iteration}")
                    
                    # Display transcription
                    st.write("**AI:s transkription:**")
                    st.write(transcription_msg["content"])
                    
                    # Display reflection
                    st.write("**AI:s reflektion:**")
                    st.write(reflection_msg["content"])
                    
                    st.divider()
                    
                    # Move to next iteration
                    iteration += 1
                    message_index += 4
                else:
                    # Skip to next message - we shouldn't have direct mode messages in history anymore
                    message_index += 1
            else:
                # Handle remaining messages
                st.write("*Ofullständig träningsiteration*")
                break
    else:
        st.write("Ingen träningshistorik än.")
