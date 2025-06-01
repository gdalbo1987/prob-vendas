import os
import joblib as jb
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Carrega modelo treinado

BASE_DIR   = os.path.dirname(__file__)  # ex: '/app' dentro do container
MODEL_PATH = os.path.join(BASE_DIR, "modelo", "prob_compra_1.pkl")

try:
    model = jb.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f'Falha ao carregar o modelo de previsão: {e}')
except:
    raise FileNotFoundError(f'Arquivo do modelo não encontrado em {MODEL_PATH}.')

# Colunas dos dados de entrada da previsão

FEATURE_COLUMNS = [
    "Idade",
    "Gênero",
    "Venda_Anual",
    "Qtde_Compras",
    "Categoria",
    "Tempo_Site",
    "Fidelidade",
    "Desconto_Utilizado"
]

# Valida dados recebidos

def valida_dados(dados: dict) -> tuple[bool, str]:

    """ 
    Verifica se todos os dados foram recebidos corretamente.
    Retorna (True, ""), se tudo estiver ok e (False, "Mensagem de Erro") se detectado algum problema.
    """

    faltantes = [col for col in FEATURE_COLUMNS if col not in dados]
    if faltantes:
        return False, f"Faltando as seguinte entradas: {faltantes}."
    return True, ""

# Roteamento da API

@app.route('/prob', methods = ['POST'], strict_slashes = False)

# Função que realiza a predição de probabilidade

def prediction():

    """
    Recebe um json com os dados de entrada, calcula as features adicionais e retorna
    a probabilidade de compra futura como:

    {'prob': 0.78}
    """

    # Verifica dados no corpo da requisição

    if not request.is_json:
        return jsonify({'Erro':'JSON no corpo da requisição não encontrado'}), 400
    
    dados_json = request.get_json()

    # Verifica as chaves dos dados recebidos

    ok, mensagem = valida_dados(dados_json)

    if not ok:
        return jsonify({'Erro': mensagem}), 400
    
    # Monta dataframe usado para predição

    try:
        valores = {col: [dados_json[col]] for col in FEATURE_COLUMNS}
        df = pd.DataFrame(valores, columns = FEATURE_COLUMNS)
    except Exception as e:
        return jsonify({'Erro': f'Falha na montagem do dataframe: {str(e)}'}), 500
    
    # Constroi duas features adicionais para previsão

    df['Tempo_Compra'] = df['Tempo_Site']/(df['Qtde_Compras'] + 1)
    df['Valor_Compra'] = df['Venda_Anual']/(df['Qtde_Compras'] + 1)

    # Gera previsão de probabilidade de compra

    try:
        pred = model.predict_proba(df)
        prob = pred[0][1]
    except AttributeError:
        return jsonify({'Erro':'Não foi possível realizar a previsão.'}), 500
    except Exception as e:
        return jsonify({'Erro':f'Erro durante a predição: {str(e)}'}), 500

    # Retorna o resultado da previsão
    pct = round(prob, 2) * 100
    return jsonify({'prob': pct}), 200

# Execução no servidor

if __name__=='__main__':
    port = int(os.environ.get("PORT", 7860))
    app.run(host = '0.0.0.0', port = port, debug = True)






