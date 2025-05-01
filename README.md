# Product Return Risk Score Prediction

Bu proje, müşteri sipariş geçmişi ve davranış kalıplarına dayanarak ürün iade riskini tahmin eden bir derin öğrenme modeli uygular.

## Features

- TensorFlow kullanılarak geliştirilmiş derin öğrenme modeli
- Dengesiz verilerle başa çıkmak için maliyet duyarlı (cost-sensitive) öğrenme
- SHAP tabanlı model açıklamaları
- Sipariş ve müşteri verilerinden özellik mühendisliği
- Model performansı ve özellik öneminin görselleştirilmesi

## Project Structure

- `train_model.py`: Veri ön işleme, model eğitimi ve değerlendirmesi için ana betik
- `requirements.txt`: Proje bağımlılıklarını içeren dosya
- `training_history.png`: Modelin eğitim sürecinin görselleştirilmesi
- `shap_summary.png`: Özellik önem sıralamasının görselleştirilmesi

## Features Used

1. Sipariş düzeyinde özellikler:
   - Siparişin toplam tutarı
   - Ortalama indirim
   - Toplam ürün adedi

2. Müşteri düzeyinde özellikler:
   - Toplam sipariş sayısı
   - Ortalama sipariş tutarı
   - Toplam harcama
   - Tüm siparişlerdeki ortalama indirim

## Target Variable

Hedef değişken (IsRisky) aşağıdaki koşullara göre oluşturulmuştur:
- Yüksek indirim (%15'in üzerinde)
- Düşük harcama (sipariş tutarlarının en düşük %25'lik diliminde yer alma)

## Setup

1. Sanal ortam oluşturun:
```bash
python -m venv venv
source venv/bin/activate  # Windows için: venv\Scripts\activate
