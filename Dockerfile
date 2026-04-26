# Gunakan OS Linux mini yang sudah terinstall Python 3.11
FROM python:3.11-slim

# Bikin folder kerja di dalam server Koyeb nanti
WORKDIR /code

# Salin daftar pustaka dan install
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Salin seluruh kode aplikasi kita
COPY ./app /code/app

# Perintah wajib untuk menghidupkan server FastAPI di cloud
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]