import gradio as gr
from transformers import pipeline

print("🚀 Akıllı sohbet asistanı yükleniyor, lütfen bekleyin...")

# Kararlı ve hızlı bir metin üretim modeli yüklüyoruz
generator = pipeline("text-generation", model="distilgpt2")

def ai_cevapla(mesaj, gecmis):
    if not mesaj or not mesaj.strip():
        return "Lütfen bir şeyler yaz."
    
    # Sohbet formatını belirleyelim
    prompt = f"Human: {mesaj}\nAI:"
    
    try:
        # Yapay zekanın yanıt üretmesi
        sonuc = generator(
            prompt, 
            max_new_tokens=60, 
            do_sample=True, 
            temperature=0.7,
            pad_token_id=50256
        )
        
        uretilen_metin = sonuc[0]['generated_text']
        
        # Sadece AI'ın verdiği cevabı ayıklayalım
        if "AI:" in uretilen_metin:
            cevap = uretilen_metin.split("AI:")[-1].strip()
            # Eğer alt satıra geçerse ilk cümleyi al
            cevap = cevap.split("\n")[0]
        else:
            cevap = "Anladım, başka nasıl yardımcı olabilirim?"
            
        return cevap if cevap else "Bu konuda söyleyecek başka bir şeyim yok."
        
    except Exception as e:
        return f"Bir hata oluştu: {str(e)}"

# Gradio'nun yerleşik, hatasız sohbet arayüzü
demo = gr.ChatInterface(
    fn=ai_cevapla,
    title="🤖 GitHub Gerçek Zamanlı Yapay Zeka Asistanı",
    description="Bu arayüz mesajlarını tekrarlamaz, doğrudan yapay zeka mantığıyla yanıt üretir.",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
