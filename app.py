import gradio as gr
from transformers import pipeline

print("🚀 Sohbet yapay zekası yükleniyor...")

# Microsoft'un sohbet modelini yüklüyoruz
chatbot_pipeline = pipeline("text-generation", model="microsoft/DialoGPT-medium")

def ai_sohbet(kullanici_mesaji, tarihsel_gecmis):
    if not kullanici_mesaji or not kullanici_mesaji.strip():
        return "", tarihsel_gecmis
    
    if tarihsel_gecmis is None:
        tarihsel_gecmis = []
        
    try:
        # Modele yanıt üretme
        cevap = chatbot_pipeline(kullanici_mesaji, max_length=100, num_return_sequences=1)
        bot_yaniti = cevap[0]['generated_text']
        
        # Güncel Gradio Chatbot formatı: liste içinde sözlükler veya listeler (role ve content)
        tarihsel_gecmis.append({"role": "user", "content": kullanici_mesaji})
        tarihsel_gecmis.append({"role": "assistant", "content": bot_yaniti})
        
        return "", tarihsel_gecmis
    except Exception as e:
        tarihsel_gecmis.append({"role": "user", "content": kullanici_mesaji})
        tarihsel_gecmis.append({"role": "assistant", "content": f"Bir hata oluştu: {str(e)}"})
        return "", tarihsel_gecmis

# Gradio Arayüzü
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 GitHub Akıllı Sohbet Asistanı")
    gr.Markdown("Bu arayüz üzerinden yapay zeka ile doğrudan yazışabilir ve sohbet edebilirsin.")
    
    chatbot_arayuz = gr.Chatbot(label="Sohbet Geçmişi, type=\"messages\"")
    mesaj_kutusu = gr.Textbox(label="Mesajını yaz...", placeholder="Merhaba, nasılsın?")
    
    gonder_btn = gr.Button("Gönder", variant="primary")
    
    gonder_btn.click(ai_sohbet, inputs=[mesaj_kutusu, chatbot_arayuz], outputs=[mesaj_kutusu, chatbot_arayuz])
    mesaj_kutusu.submit(ai_sohbet, inputs=[mesaj_kutusu, chatbot_arayuz], outputs=[mesaj_kutusu, chatbot_arayuz])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
