import os
import joblib as jb
import pandas as pd
from flask import Flask, request, jsonify, make_response

app = Flask(__name__)

# ──────────────── Carrega o modelo ────────────────
BASE_DIR   = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "modelo", "prob_compra_1.pkl")

try:
    model = jb.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Falha ao carregar o modelo: {e}")

# ──────────────── Colunas esperadas ────────────────
FEATURE_COLUMNS = [
    "Idade", "Gênero", "Venda_Anual", "Qtde_Compras",
    "Categoria", "Tempo_Site", "Fidelidade", "Desconto_Utilizado"
]

def valida_dados(dados: dict) -> tuple[bool, str]:
    faltantes = [c for c in FEATURE_COLUMNS if c not in dados]
    return (False, f"Faltando entradas: {faltantes}") if faltantes else (True, "")

# ──────────────── Função utilitária para CORS ────────────────
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"]  = "*"            # ou seu domínio
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return resp

# ──────────────── Endpoint ────────────────
@app.route("/prob", methods=["POST", "OPTIONS"], strict_slashes=False)
def prediction():
    # 1) Pre-flight (browser envia OPTIONS antes do POST)
    if request.method == "OPTIONS":
        return add_cors_headers(make_response("", 204))  # sem corpo

    # 2) POST real
    if not request.is_json:
        return add_cors_headers(jsonify({"Erro": "JSON não encontrado"})), 400

    dados_json = request.get_json()
    ok, msg = valida_dados(dados_json)
    if not ok:
        return add_cors_headers(jsonify({"Erro": msg})), 400

    try:
        df = pd.DataFrame({c: [dados_json[c]] for c in FEATURE_COLUMNS})
    except Exception as e:
        return add_cors_headers(jsonify({"Erro": f"DataFrame: {e}"})), 500

    df["Tempo_Compra"] = df["Tempo_Site"] / (df["Qtde_Compras"] + 1)
    df["Valor_Compra"] = df["Venda_Anual"] / (df["Qtde_Compras"] + 1)

    try:
        prob = model.predict_proba(df)[0][1]          # retorno de 0-1
    except Exception as e:
        return add_cors_headers(jsonify({"Erro": f"Predição: {e}"})), 500

    pct = round(prob, 2) * 100                        # 0-100 %
    return add_cors_headers(jsonify({"prob": pct})), 200

# ──────────────── Main ────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
