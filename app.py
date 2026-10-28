import gradio as gr
from transformers import pipeline

print("🚀 Sohbet yapay zekası yükleniyor...")

# Microsoft'un geliştirdiği sohbet (conversational) modelini yüklüyoruz
chatbot_pipeline = pipeline("text-generation", model="microsoft/DialoGPT-medium")

def ai_sohbet(kullanici_mesaji, tarihsel_gecmis=[]):
    if not kullanici_mesaji.strip():
        return "", tarihsel_gecmis
    
    try:
        # Modele girdi oluşturma ve yanıt üretme
        cevap = chatbot_pipeline(kullanici_mesaji, max_length=100, num_return_sequences=1)
        bot_yaniti = cevap[0]['generated_text']
        
        # Geçmiş sohbeti listeye ekle
        tarihsel_gecmis.append((kullanici_mesaji, bot_yaniti))
        return "", tarihsel_gecmis
    except Exception as e:
        tarihsel_gecmis.append((kullanici_mesaji, f"Bir hata oluştu: {str(e)}"))
        return "", tarihsel_gecmis

# Gradio Arayüzü
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 GitHub Akıllı Sohbet Asistanı")
    gr.Markdown("Bu arayüz üzerinden yapay zeka ile doğrudan yazışabilir ve sohbet edebilirsin.")
    
    chatbot_arayuz = gr.Chatbot(label="Sohbet Geçmişi")
    mesaj_kutusu = gr.Textbox(label="Mesajını yaz...", placeholder="Merhaba, nasılsın?")
    
    gonder_btn = gr.Button("Gönder", variant="primary")
    
    gonder_btn.click(ai_sohbet, inputs=[mesaj_kutusu, chatbot_arayuz], outputs=[mesaj_kutusu, chatbot_arayuz])
    mesaj_kutusu.submit(ai_sohbet, inputs=[mesaj_kutusu, chatbot_arayuz], outputs=[mesaj_kutusu, chatbot_arayuz])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
