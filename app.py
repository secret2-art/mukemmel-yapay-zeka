import gradio as gr
import google.generativeai as genai

# Google Gemini API Yapılandırması
API_KEY = "AQ.Ab8RN6IQaLJiSwjH3z1sPvf8N7bTepmX-fnlxLmW8apu0pKaxQ"
genai.configure(api_key=API_KEY)

# Gemini Modelini Tanımlıyoruz
model = genai.GenerativeModel('gemini-1.5-flash')

def sohbet_et(mesaj, gecmis):
    if not mesaj or not mesaj.strip():
        return ""
    
    try:
        # Geçmişi Gemini'nin anlayacağı formata çeviriyoruz
        gemini_gecmis = []
        if gecmis:
            for kullanıcı_mesajı, ai_yanıtı in gecmis:
                gemini_gecmis.append({"role": "user", "parts": [kullanıcı_mesajı]})
                gemini_gecmis.append({"role": "model", "parts": [ai_yanıtı]})
        
        # Sohbet oturumunu başlatıp yanıt alıyoruz
        chat = model.start_chat(history=gemini_gecmis)
        yanit = chat.send_message(mesaj)
        return yanit.text
        
    except Exception as e:
        return f"Bir hata oluştu: {str(e)}"

# Gradio'nun en güncel ve kararlı ChatInterface bileşeni
demo = gr.ChatInterface(
    fn=sohbet_et,
    title="🧠 GitHub Gemini Pro Yapay Zeka Asistanı",
    description="Doğrudan Google altyapısıyla çalışır. Sorduğun her soruya uzun, detaylı ve mantıklı yanıtlar verir.",
    textbox=gr.Textbox(placeholder="Yapay zekaya dilediğini sor...", container=False, scale=7),
    theme="soft",
    retry_btn="Tekrar Dene",
    undo_btn="Geri Al",
    clear_btn="Temizle"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
