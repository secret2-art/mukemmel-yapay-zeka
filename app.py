import gradio as gr
import google.generativeai as genai

# 1. Google Gemini API Yapılandırması (Anahtarın doğrudan eklenmiştir)
API_KEY = "AQ.Ab8RN6IQaLJiSwjH3z1sPvf8N7bTepmX-fnlxLmW8apu0pKaxQ"

genai.configure(api_key=API_KEY)

# 2. Dünyanın en hızlı ve güçlü modellerinden biri
model = genai.GenerativeModel('gemini-1.5-flash')

# Sohbet hafızasını başlatıyoruz
chat_session = model.start_chat(history=[])

def yapay_zeka_cevapla(kullanici_mesaji, tarihsel_gecmis):
    if not kullanici_mesaji or not kullanici_mesaji.strip():
        return "", tarihsel_gecmis
    
    if tarihsel_gecmis is None:
        tarihsel_gecmis = []
        
    try:
        # Gemini modeline mesajı gönderip yanıt alıyoruz
        response = chat_session.send_message(kullanici_mesaji)
        bot_yaniti = response.text
        
        # Gradio modern sohbet formatına ekleme
        tarihsel_gecmis.append({"role": "user", "content": kullanici_mesaji})
        tarihsel_gecmis.append({"role": "assistant", "content": bot_yaniti})
        
        return "", tarihsel_gecmis
    
    except Exception as e:
        hata_mesaji = f"⚠️ Hata: {str(e)}"
        tarihsel_gecmis.append({"role": "user", "content": kullanici_mesaji})
        tarihsel_gecmis.append({"role": "assistant", "content": hata_mesaji})
        return "", tarihsel_gecmis

# Gradio ile Şık ve Profesyonel Arayüz
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Gerçek ve Sınırsız GitHub Yapay Zeka Asistanı")
    gr.Markdown("Bu asistan doğrudan **Google Gemini** altyapısıyla çalışır ve sorduğun her soruya eksiksiz, uzun ve detaylı yanıtlar verir.")
    
    chatbot_arayuz = gr.Chatbot(label="Sohbet Paneli", type="messages", height=500)
    
    with gr.Row():
        mesaj_kutusu = gr.Textbox(
            label="Yapay zekaya ne sormak istiyorsun?", 
            placeholder="Örn: Bana uzun bir Python kodu yaz ve açıkla...", 
            scale=4
        )
        gonder_btn = gr.Button("Gönder", variant="primary", scale=1)
    
    gonder_btn.click(yapay_zeka_cevapla, inputs=[mesaj_kutusu, chatbot_arayuz], outputs=[mesaj_kutusu, chatbot_arayuz])
    mesaj_kutusu.submit(yapay_zeka_cevapla, inputs=[mesaj_kutusu, chatbot_arayuz], outputs=[mesaj_kutusu, chatbot_arayuz])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
