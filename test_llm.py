from app.services.llm_service import generate_stock_analysis

ticker_saham = "BBCA.JK"
print(f"Menarik data ML dan AI untuk {ticker_saham}...\n")

hasil = generate_stock_analysis(ticker_saham)

print("=========================================")
print("📊 HASIL ANALISIS LLAMA 3.2 + XGBOOST")
print("=========================================")
print(hasil)