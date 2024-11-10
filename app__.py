from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import logging
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage
from groq import Groq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

app = Flask(__name__)
CORS(app)  # Ativa o CORS para permitir requisições do frontend

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Inicialize o cliente Groq com a chave da API
api_key = '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'  # Substitua por sua chave de API real
client = Groq(api_key=api_key)

# Diretórios para salvar a memória da conversa e o banco vetorial
MEMORY_DIR = 'conversa_memoria'
VECTORSTORE_DIR = 'vectorstore_data'

# Cria os diretórios se não existirem
os.makedirs(MEMORY_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# Inicializa o modelo de embeddings e o banco vetorial com persistência
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(embedding_function=embedding_model, persist_directory=VECTORSTORE_DIR)

# Prompt do sistema para guiar a conversa
system_prompt = (
    "Você é um assistente que ajuda usuários a explorar e descobrir novas oportunidades de desenvolvimento profissional, "
    "inclusive aquelas que podem não ter considerado antes. Conduza a conversa para coletar as seguintes informações: "
    "1. Nível de escolaridade do usuário. "
    "2. Área de trabalho atual e satisfação com o trabalho. "
    "3. Objetivo profissional, se houver, e abertura para novas áreas ou funções. "
    "4. Cursos, treinamentos desejados e áreas de interesse. "
    "5. Preferência por oportunidades presenciais ou virtuais. "
    "6. Limitações de tempo ou recursos que possam afetar o aproveitamento das oportunidades. "
    "Conduza a conversa de forma gentil, incentivando o usuário a considerar alternativas que possam oferecer melhores condições de vida ou novas perspectivas. "
    "Evite sugerir qualquer caminho específico antes de ter todas as informações, mas deixe o usuário aberto a novas possibilidades que ele possa desconhecer. "
    "Finalize a conversa dizendo: 'Obrigado pelas informações, vou analisar as oportunidades que se encaixam no seu perfil!' quando todos os dados tiverem sido coletados."
)

# Função para carregar a memória da conversa
def carregar_memoria(session_id):
    memory_file = os.path.join(MEMORY_DIR, f'memoria_{session_id}.json')
    logging.info(f"Carregando memória da conversa para a sessão {session_id}...")
    if os.path.exists(memory_file):
        with open(memory_file, 'r', encoding='utf-8') as f:
            messages_data = json.load(f)
            messages = []
            for msg in messages_data:
                if msg['type'] == 'human':
                    messages.append(HumanMessage(content=msg['content']))
                elif msg['type'] == 'ai':
                    messages.append(AIMessage(content=msg['content']))
            logging.info("Memória carregada com sucesso.")
            return messages
    else:
        logging.info("Nenhuma memória anterior encontrada. Iniciando nova conversa.")
        return []

# Função para salvar a memória da conversa
def salvar_memoria(messages, session_id):
    memory_file = os.path.join(MEMORY_DIR, f'memoria_{session_id}.json')
    logging.info(f"Salvando memória da conversa para a sessão {session_id}...")
    messages_data = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            msg_type = 'human'
        elif isinstance(msg, AIMessage):
            msg_type = 'ai'
        else:
            continue
        messages_data.append({'type': msg_type, 'content': msg.content})
    with open(memory_file, 'w', encoding='utf-8') as f:
        json.dump(messages_data, f, ensure_ascii=False, indent=2)
    logging.info("Memória salva com sucesso.")

# Função para gerar respostas com o modelo Groq
def gerar_resposta_groq(messages):
    logging.info("Gerando resposta do modelo Groq...")
    # Construindo as mensagens para o modelo
    model_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        if isinstance(msg, HumanMessage):
            model_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            model_messages.append({"role": "assistant", "content": msg.content})
    # Chamando a API do modelo
    response = client.chat.completions.create(
        model="llama-3.2-90b-text-preview",
        messages=model_messages,
        temperature=0.3,
        max_tokens=300,
        top_p=1,
        stream=False,
        stop=None,
    )
    resposta = response.choices[0].message.content.strip()
    logging.info("Resposta gerada com sucesso.")
    return resposta

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

@app.route('/mensagem', methods=['POST'])
def mensagem():
    dados = request.get_json()
    mensagem_usuario = dados.get('mensagem')
    session_id = dados.get('session_id')

    # Carrega a memória da conversa
    messages = carregar_memoria(session_id)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    memory.chat_memory.messages = messages

    # Adiciona a mensagem do usuário na memória
    memory.chat_memory.add_user_message(mensagem_usuario)

    # Gera a resposta do chatbot
    resposta_chatbot = gerar_resposta_groq(memory.chat_memory.messages)

    # Adiciona a resposta do chatbot na memória
    memory.chat_memory.add_ai_message(resposta_chatbot)

    # Salva a memória da conversa
    salvar_memoria(memory.chat_memory.messages, session_id)

    # Retorna a resposta para o front-end
    return jsonify({'resposta': resposta_chatbot})

# Novo endpoint para enviar as mensagens anteriores
@app.route('/conversa', methods=['POST'])
def conversa():
    dados = request.get_json()
    session_id = dados.get('session_id')
    messages = carregar_memoria(session_id)
    # Convertendo as mensagens para um formato serializável
    messages_data = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            msg_role = 'user'
        elif isinstance(msg, AIMessage):
            msg_role = 'bot'
        else:
            continue
        messages_data.append({'role': msg_role, 'content': msg.content})
    return jsonify({'messages': messages_data})

if __name__ == '__main__':
    app.run(debug=True)
