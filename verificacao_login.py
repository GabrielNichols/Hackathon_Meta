from flask import Flask, request, jsonify
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)  # Ativa o CORS para permitir requisições do frontend

# Função de verificação de login
@app.route('/login', methods=['POST'])
def login():
    dados = request.get_json()
    email = dados.get('email')
    senha = dados.get('senha')

    # Carrega os dados de login do arquivo JSON
    with open('usuarios.json') as f:
        usuarios = json.load(f)

    # Verifica se as credenciais estão corretas
    if email in usuarios and usuarios[email] == senha:
        return jsonify({'sucesso': True, 'mensagem': 'Login bem-sucedido!'})
    else:
        return jsonify({'sucesso': False, 'mensagem': 'Credenciais inválidas'})

if __name__ == '__main__':
    app.run(debug=True)
