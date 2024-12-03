import streamlit as st
import openai
from gtts import gTTS
import tempfile
from transformers import pipeline
import whisper
import os
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Translation Models
translation_models = {
    "en": {"to_en": None, "from_en": None},  # English doesn't need translation
    "hi": {"to_en": "Helsinki-NLP/opus-mt-hi-en", "from_en": "Helsinki-NLP/opus-mt-en-hi"},
    "es": {"to_en": "Helsinki-NLP/opus-mt-es-en", "from_en": "Helsinki-NLP/opus-mt-en-es"},
    "pt": {"to_en": "Helsinki-NLP/opus-mt-tc-big-en-pt", "from_en": "Helsinki-NLP/opus-mt-en-ROMANCE"},
    "ar": {"to_en": "Helsinki-NLP/opus-mt-ar-en", "from_en": "Helsinki-NLP/opus-mt-en-ar"},
    "fr": {"to_en": "Helsinki-NLP/opus-mt-fr-en", "from_en": "Helsinki-NLP/opus-mt-en-fr"},
    "de": {"to_en": "Helsinki-NLP/opus-mt-de-en", "from_en": "Helsinki-NLP/opus-mt-en-de"},
}

# gTTS Language Mapping
gtts_language_mapping = {
    "en": "en",  # English
    "hi": "hi",  # Hindi
    "es": "es",  # Spanish
    "pt": "pt",  # Portuguese
    "ar": "ar",  # Arabic
    "fr": "fr",  # French
    "de": "de",  # German
}

# Farewell Keywords
farewell_keywords = {
    "en": ["bye", "goodbye", "exit"],
    "hi": ["अलविदा", "विदा", "बाय"],
    "es": ["adiós", "chau", "salir"],
    "pt": ["tchau", "adeus", "sair"],
    "ar": ["مع السلامة", "وداعا", "خروج"],
    "fr": ["au revoir", "adieu", "sortir"],
    "de": ["auf wiedersehen", "tschüss", "verlassen"]
}

@st.cache(allow_output_mutation=True)
def load_translation_pipeline(model_name):
    """Load the MarianMT pipeline for the specified model."""
    try:
        return pipeline("translation", model=model_name)
    except Exception as e:
        st.error(f"Error loading translation pipeline: {str(e)}")
        return None

def translate_text(text, source_lang, target_lang):
    """Translate text using Helsinki-NLP models."""
    if source_lang == target_lang:
        return text  # No translation needed
    try:
        if target_lang == "en":
            model_name = translation_models[source_lang]["to_en"]
        else:
            model_name = translation_models[target_lang]["from_en"]
        translation_pipe = load_translation_pipeline(model_name)
        if translation_pipe:
            translated_text = translation_pipe(text)[0]["translation_text"]
            return translated_text
        else:
            raise ValueError("Translation pipeline not available.")
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
    return text

def generate_chatgpt_response(prompt):
    """Generate a response using GPT-3.5."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful, positive AI assistant. Respond in a kind and engaging way, and end the conversation politely if the user says 'bye'."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
    return None

def text_to_speech(text, lang="en"):
    """Convert text to speech using gTTS."""
    try:
        gtts_lang = gtts_language_mapping.get(lang, "en")
        tts = gTTS(text=text, lang=gtts_lang)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        return temp_file.name
    except Exception as e:
        st.error(f"Text-to-Speech error: {str(e)}")
    return None

def transcribe_audio(file_path):
    """Transcribe audio using OpenAI Whisper."""
    try:
        model = whisper.load_model("base")
        result = model.transcribe(file_path)
        return result["text"]
    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        return None

def capture_real_time_audio():
    """Capture real-time audio from the microphone."""
    import speech_recognition as sr
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Please speak now.")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            print("Audio captured.")
            temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            with open(temp_audio_path, "wb") as f:
                f.write(audio.get_wav_data())
            return temp_audio_path
        except sr.WaitTimeoutError:
            st.error("Listening timed out. Please try again.")
        except Exception as e:
            st.error(f"Error capturing audio: {str(e)}")
    return None

# Streamlit UI
st.title("Multilingual Voice Assistant")

# Conversation Log
if "conversation_log" not in st.session_state:
    st.session_state.conversation_log = []

# Language Selection
selected_language = st.selectbox(
    "Select Your Language:",
    options=["English", "Hindi", "Spanish", "Portuguese", "Arabic", "French", "German"],
    index=0
)
selected_language_code = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "Portuguese": "pt",
    "Arabic": "ar",
    "French": "fr",
    "German": "de"
}[selected_language]

# Real-Time Voice Input
if st.button("Speak"):
    audio_path = capture_real_time_audio()
    if audio_path:
        # Transcribe Audio
        transcribed_text = transcribe_audio(audio_path)
        if transcribed_text:
            st.write(f"**Recognized Input:** {transcribed_text}")
            st.write(f"**Selected Language:** {selected_language} ({selected_language_code.upper()})")

            # Check for Farewell Keywords
            if any(word.lower() in transcribed_text.lower() for word in farewell_keywords[selected_language_code]):
                farewell_message = translate_text("Goodbye! Have a nice day!", "en", selected_language_code)
                st.write(f"**Assistant:** {farewell_message}")
                audio_file = text_to_speech(farewell_message, lang=selected_language_code)
                if audio_file:
                    st.audio(audio_file, format="audio/mp3")
                st.stop()

            # Translate Input to English
            if selected_language_code != "en":
                translated_input = translate_text(transcribed_text, selected_language_code, "en")
                st.write(f"**Translated Input to English:** {translated_input}")
            else:
                translated_input = transcribed_text

            # Generate GPT Response
            gpt_response = generate_chatgpt_response(translated_input)
            st.write(f"**GPT Response in English:** {gpt_response}")

            # Translate GPT Response Back to Selected Language
            if selected_language_code != "en":
                translated_response = translate_text(gpt_response, "en", selected_language_code)
                st.write(f"**Response in {selected_language} ({selected_language_code.upper()}):** {translated_response}")

                # Add to Conversation Log
                st.session_state.conversation_log.append((transcribed_text, translated_response))

                # Convert Translated Response to Speech
                audio_file = text_to_speech(translated_response, lang=selected_language_code)
                if audio_file:
                    st.audio(audio_file, format="audio/mp3")
            else:
                st.session_state.conversation_log.append((transcribed_text, gpt_response))
                audio_file = text_to_speech(gpt_response, lang="en")
                if audio_file:
                    st.audio(audio_file, format="audio/mp3")

# Display Conversation Log
st.write("### Conversation Log:")
for user_input, assistant_response in st.session_state.conversation_log:
    st.write(f"**User:** {user_input}")
    st.write(f"**Assistant:** {assistant_response}")