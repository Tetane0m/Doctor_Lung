# استخدام صورة Python 3.10 كقاعدة
FROM python:3.10

# تحديد مجلد العمل في الحاوية
WORKDIR /app

# نسخ كل ملفات المشروع إلى الحاوية
COPY . /app

# تثبيت المكتبات المطلوبة
RUN pip install --no-cache-dir -r requirements.txt

# تحديد الأمر الذي سيشغل عند تشغيل الحاوية
CMD ["python", "bot.py"]
