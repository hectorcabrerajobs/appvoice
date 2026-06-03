import io
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile
import speech_recognition as sr
import streamlit as st
from streamlit_mic_recorder import mic_recorder

# --- Configuración de la página ---
st.set_page_config(page_title="Detector de Palabras", page_icon="🎙️")

st.title("Detector de Palabra Perro + Espectrograma")
st.write(
    "Haz clic en el botón para empezar a grabar, habla y luego detén la grabación."
)


# --- Grabador ---
audio_output = mic_recorder(
    start_prompt="🔴 Iniciar Grabación",
    stop_prompt="⏹️ Detener y Procesar",
    just_once=False,
    use_container_width=True,
    key="grabador_voz",
)

if audio_output is not None:

    audio_bytes = audio_output["bytes"]

    st.write("Formato detectado:", audio_output.get("format", "desconocido"))
    st.audio(audio_bytes)

    with st.spinner("Procesando audio..."):

        reconocedor = sr.Recognizer()

        try:
            # Leer el audio recibido
            audio_original = AudioSegment.from_file(io.BytesIO(audio_bytes))

            # Convertir a WAV PCM mono 16 kHz
            audio_convertido = audio_original.set_channels(1).set_frame_rate(
                16000
            )

            wav_buffer = io.BytesIO()
            audio_convertido.export(wav_buffer, format="wav", codec="pcm_s16le")

            wav_buffer.seek(0)

            # --- NUEVA SECCIÓN: Generación del Espectrograma ---
            # Leemos la tasa de muestreo y los datos numéricos del buffer WAV
            tasa_muestreo, datos_señal = wavfile.read(wav_buffer)

            # Reiniciamos el puntero del buffer para que SpeechRecognition pueda leerlo después
            wav_buffer.seek(0)

            # SpeechRecognition ahora sí puede leerlo
            with sr.AudioFile(wav_buffer) as origen:
                datos_audio = reconocedor.record(origen)

            texto_detectado = reconocedor.recognize_google(
                datos_audio, language="es-ES"
            )

            st.success("Transcripción realizada")
            st.info(f"Texto detectado: {texto_detectado}")

            if "perro" in texto_detectado.lower():
                st.success("🐶 ¡Has dicho 'perro'!")
            else:
                st.warning("No se detectó la palabra 'perro'.")

            # --- Mostrar el Espectrograma en la interfaz ---
            st.write("---")
            st.subheader("Espectrograma del Audio Original")

            # Creamos la figura de Matplotlib
            fig, ax = plt.subplots(figsize=(10, 4))

            # Graficamos el espectrograma usando la función specgram
            # cmap='viridis' o 'magma' suelen ser muy visuales para esto
            valores_p, frecuencias, tiempos, im = ax.specgram(
                datos_señal, Fs=tasa_muestreo, cmap="magma"
            )

            ax.set_xlabel("Tiempo (segundos)")
            ax.set_ylabel("Frecuencia (Hz)")
            fig.colorbar(im, ax=ax, label="Intensidad (dB)")

            # Enviamos el gráfico de Matplotlib directamente a Streamlit
            st.pyplot(fig)

        except sr.UnknownValueError:
            st.error("No se pudo entender el audio.")

        except sr.RequestError as e:
            st.error(f"Error de conexión con Google Speech: {e}")

        except Exception as e:
            st.error(f"Error procesando audio: {e}")
            st.exception(e)
