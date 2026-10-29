import gradio as gr
import google.generativeai as genai

# Google Gemini API Yapılandırması
API_KEY = "AQ.Ab8RN6IQaLJiSwjH3z1sPvf8N7bTepmX-fnlxLmW8apu0pKaxQ"
genai.configure(api_key=API_KEY)

# Gemini Modelini Ayarlıyoruz
model = genai.GenerativeModel('gemini-1.5-flash')
chat_session = model.start_chat(history=[])

def yapay_zeka_cevapla(kullanici_mesaji, tarihsel_gecmis):
    if not kullanici_mesaji or not kullanici_mesaji.strip():
        return "", tarihsel_gecmis
    
    if tarihsel_gecmis is None:
        tarihsel_gecmis = []
        
    try:
        response = chat_session.send_message(kullanici_mesaji)
        bot_yaniti = response.text
        
        tarihsel_gecmis.append((kullanici_mesaji, bot_yaniti))
        return "", tarihsel_gecmis
    except Exception as e:
        hata_mesaji = f"⚠️ Hata: {str(e)}"
        tarihsel_gecmis.append((kullanici_mesaji, hata_mesaji))
        return "", tarihsel_gecmis

# En temel ve kararlı Gradio Sohbet Yapısı
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Gerçek Gemini Yapay Zeka Asistanı")
    
    chatbot_arayuz = gr.Chatbot(height=450)
    mesaj_kutusu = gr.Textbox(label="Mesajını yaz...", placeholder="Buraya yazıp Enter'a bas...")
    
    mesaj_kutusu.submit(yapay_zeka_cevapla, inputs=[mesaj_kutusu, chatbot_arayuz], outputs=[mesaj_kutusu, chatbot_arayuz])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
