import gradio as gr
from transformers import pipeline
from PIL import Image
import requests

print("🚀 Yapay zeka modelleri yükleniyor, lütfen bekleyin...")

# 1. Metin Duygu Analizi Modeli (Türkçe ve İngilizce destekler)
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# 2. Görsel Sınıflandırma Modeli (Görselde ne olduğunu anlar)
image_classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

print("✅ Tüm yapay zeka modelleri başarıyla yüklendi!")

# Fonksiyon 1: Duygu Analizi
def duygu_analizi_yap(metin):
    if not metin or not metin.strip():
        return "⚠️ Lütfen analiz edilecek bir metin girin."
    
    sonuc = sentiment_analyzer(metin)[0]
    etiket = sonuc['label']
    skor = round(sonuc['score'] * 100, 2)
    
    if etiket == "POSITIVE":
        return f"😊 Pozitif Duygu (Güven: %{skor})"
    else:
        return f"😔 Negatif Duygu (Güven: %{skor})"

# Fonksiyon 2: Görsel Tanıma
def gorsel_tani_url(url):
    if not url or not url.strip():
        return "⚠️ Lütfen geçerli bir görsel URL'si girin."
    try:
        image = Image.open(requests.get(url, stream=True).raw)
        sonuclar = image_classifier(image)
        
        # En yüksek tahminleri düzenle
        cikti = "🎯 Yapay Zeka Tahminleri:\n\n"
        for s in sonuclar[:3]: # En iyi 3 tahmini al
            oran = round(s['score'] * 100, 2)
            cikti += f"• {s['label']} (Oran: %{oran})\n"
        return cikti
    except Exception as e:
        return f"❌ Görsel yüklenirken hata oluştu: {str(e)}"

# Gradio ile Kusursuz Web Arayüzü Oluşturma
with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown("# 🧠 Kusursuz Çalışan GitHub Yapay Zeka Asistanı")
    gr.Markdown("Bu uygulama, son teknoloji **Hugging Face Transformers** altyapısını kullanarak metin analizi ve görsel tanıma yapar.")
    
    with gr.Tab("📝 Metin Duygu Analizi"):
        metin_input = gr.Textbox(label="Bir cümle yazın (İngilizce/Türkçe destekli):", placeholder="Örn: I love using artificial intelligence!")
        metin_output = gr.Textbox(label="Analiz Sonucu:")
        metin_btn = gr.Button("Duyguyu Analiz Et", variant="primary")
        metin_btn.click(duygu_analizi_yap, inputs=metin_input, outputs=metin_output)
        
    with gr.Tab("🖼️ Görsel Tanıma (Image Classification)"):
        url_input = gr.Textbox(label="Görsel URL'si Girin:", placeholder="http://images.cocodataset.org/val2017/000000039769.jpg")
        url_output = gr.Textbox(label="Yapay Zeka Analiz Raporu:")
        url_btn = gr.Button("Görseli Analiz Et", variant="primary")
        url_btn.click(gorsel_tani_url, inputs=url_input, outputs=url_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
