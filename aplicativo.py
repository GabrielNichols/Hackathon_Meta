from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import logging
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage
from groq import Groq
from langchain_mongodb import MongoDBAtlasVectorSearch
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import datetime
import urllib.parse
import cohere
from langchain.embeddings.base import Embeddings
import subprocess
import sys
import json

# Load environment variables from .env file
load_dotenv()

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)  # Initialize CORS to allow cross-origin requests

# Initialize the Groq client with the API key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("A chave de API do Groq não foi encontrada. Verifique se está definida no arquivo .env.")
client = Groq(api_key=api_key)

# Initialize the Cohere API key
cohere_api_key = os.getenv("COHERE_API_KEY")
if not cohere_api_key:
    raise ValueError("A chave de API do Cohere não foi encontrada. Verifique se está definida no arquivo .env.")

# Create the embeddings class using Cohere API
class CohereEmbeddings(Embeddings):
    def __init__(self, api_key, model="embed-multilingual-v2.0", truncate="RIGHT"):
        self.client = cohere.Client(api_key)
        self.model = model
        self.truncate = truncate

    def embed_documents(self, texts):
        response = self.client.embed(
            texts=texts,
            model=self.model,
            truncate=self.truncate,
        )
        return response.embeddings

    def embed_query(self, text):
        response = self.client.embed(
            texts=[text],
            model=self.model,
            truncate=self.truncate,
        )
        return response.embeddings[0]

# Initialize the embeddings model with Cohere
embedding_model = CohereEmbeddings(
    api_key=cohere_api_key,
    model="embed-multilingual-v2.0",  # Ensure the model name is correct
)

# Load MongoDB credentials
MONGODB_USERNAME = os.getenv('MONGODB_USERNAME')
MONGODB_PASSWORD = os.getenv('MONGODB_PASSWORD')

if not MONGODB_USERNAME or not MONGODB_PASSWORD:
    raise ValueError("Nome de usuário ou senha do MongoDB não encontrados no arquivo .env.")

username = urllib.parse.quote_plus(MONGODB_USERNAME)
password = urllib.parse.quote_plus(MONGODB_PASSWORD)

cluster_host = 'hackathonmeta.pvjrb.mongodb.net'

uri = f"mongodb+srv://{username}:{password}@{cluster_host}/?retryWrites=true&w=majority&appName=HackathonMeta&tls=true"

client_mongo = MongoClient(uri, server_api=ServerApi('1'))

try:
    client_mongo.admin.command('ping')
    logging.info("Conexão bem-sucedida com o MongoDB.")
except Exception as e:
    logging.error(f"Erro ao conectar ao MongoDB: {e}")
    raise

db = client_mongo['DadosUsuários']
collection_contexto = db['Contexto']
collection_historico = db['HistoricoConversa']
collection_oportunidades = db['Oportunidades']  # Collection for opportunities

vectorstore = MongoDBAtlasVectorSearch(
    collection=collection_contexto,
    embedding=embedding_model,
    index_name='contexto',
    text_key='content',
    embedding_key='embedding'
)

system_prompt = (
    "Você é um assistente especializado em ajudar usuários a encontrar oportunidades de desenvolvimento profissional. "
    "Suas respostas devem ser claras e concisas, mantendo uma abordagem amigável e informativa. "
    "Interaja com o usuário de maneira bem direta e objetiva, sempre buscando coletar informações relevantes para gerar recomendações precisas. "
    "Conduza a conversa de forma estruturada para coletar as seguintes informações essenciais do usuário: "
    "Peça informações baiscas, como nome, idade, localização e outras informações relevantes. "
    "1. Nível de escolaridade atual e desejada. "
    "2. Área de trabalho atual e nível de satisfação com o emprego atual. "
    "3. Objetivos profissionais específicos a curto e longo prazo. "
    "4. Cursos, treinamentos ou certificações que o usuário deseja realizar, incluindo áreas de interesse. "
    "5. Preferência por oportunidades presenciais, online ou híbridas. "
    "6. Limitações de tempo, financeiras ou outras que possam impactar a participação em oportunidades. "
    "Durante a conversa, de algumas sugestões de oportnidades que o usuário pode se interessar. "
    "Se todas as informações forem coletadas, pergunte ao usuário se deseja receber as recomendações agora. "
    "Lembre-se: Você não gera as recomendações diretamente, mas coleta informações precisas para que os agentes do Crew AI possam processá-las eficientemente."
    "OBS: Suas mensagens devem apenas conter uma linha durante as perguntas!"
    "Não entregue a resposta como markdown, apenas o texto."
)

def load_users():
    with open('usuarios.json', 'r') as f:
        users = json.load(f)
    return users

def carregar_memoria(user_id):
    logging.info(f"Carregando memória da conversa do MongoDB para o usuário {user_id}...")
    conversa = collection_historico.find_one({'user_id': user_id})
    if conversa and 'messages' in conversa:
        messages_data = conversa['messages']
        messages = []
        for msg in messages_data:
            if msg['type'] == 'human':
                messages.append(HumanMessage(content=msg['content']))
            elif msg['type'] == 'ai':
                messages.append(AIMessage(content=msg['content']))
        logging.info("Memória carregada com sucesso.")
        return messages
    else:
        logging.info("Nenhuma memória anterior encontrada para este usuário.")
        return []

def salvar_memoria(user_id, messages):
    logging.info(f"Salvando memória da conversa no MongoDB para o usuário {user_id}...")
    messages_data = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            msg_type = 'human'
        elif isinstance(msg, AIMessage):
            msg_type = 'ai'
        else:
            continue
        messages_data.append({
            'type': msg_type,
            'content': msg.content,
            'timestamp': datetime.datetime.utcnow()
        })
    conversa = {
        'user_id': user_id,
        'messages': messages_data,
        'last_updated': datetime.datetime.utcnow()
    }
    collection_historico.update_one(
        {'user_id': user_id},
        {'$set': conversa},
        upsert=True
    )
    logging.info("Memória da conversa salva no MongoDB.")

def gerar_resposta_groq(messages):
    logging.info("Gerando resposta do modelo Groq...")
    model_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        if isinstance(msg, HumanMessage):
            model_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            model_messages.append({"role": "assistant", "content": msg.content})
    try:
        response = client.chat.completions.create(
            model="llama-3.2-90b-text-preview",
            messages=model_messages,
            temperature=0.7,
            max_tokens=820,
            top_p=1,
            stream=False,
            stop=None,
        )
        resposta = response.choices[0].message.content.strip()
        logging.info("Resposta gerada com sucesso.")
        return resposta
    except Exception as e:
        logging.error(f"Erro ao gerar resposta com o Groq: {e}")
        return "Houve um erro ao processar sua solicitação."

def armazenar_mensagem_no_vectorstore(role, content, user_id):
    if role == 'user':
        metadata = {"role": role, "user_id": user_id}
        vectorstore.add_texts([content], metadatas=[metadata])
        logging.info("Mensagem do usuário armazenada no vectorstore.")
    else:
        logging.info("Mensagem do assistente não armazenada no vectorstore.")

def validar_contexto_suficiente(messages):
    logging.info("Validando se o contexto é suficiente para gerar recomendações.")
    validation_prompt = (
        "Dada a seguinte conversa entre o assistente e o usuário:\n"
        "{conversation}\n"
        "Determine se todas as seguintes informações foram coletadas:\n"
        "1. Nível de escolaridade do usuário.\n"
        "2. Área de trabalho atual e satisfação com o trabalho.\n"
        "3. Objetivo profissional específico.\n"
        "4. Cursos ou treinamentos desejados e áreas de interesse.\n"
        "5. Preferência por oportunidades presenciais ou virtuais.\n"
        "6. Limitações de tempo ou recursos que possam afetar o aproveitamento das oportunidades.\n"
        "Se todas essas informações foram coletadas, responda 'Sim'. Caso contrário, responda 'Não'."
    )
    conversation_text = ""
    for msg in messages:
        role = "Assistente" if isinstance(msg, AIMessage) else "Usuário"
        conversation_text += f"{role}: {msg.content}\n"
    formatted_prompt = validation_prompt.format(conversation=conversation_text)
    try:
        response = client.chat.completions.create(
            model="llama-3.2-90b-text-preview",
            messages=[{"role": "system", "content": formatted_prompt}],
            temperature=0.0,
            max_tokens=10,
            top_p=1,
            stream=False,
            stop=None,
        )
        answer = response.choices[0].message.content.strip().lower()
        logging.info(f"Resposta da validação: {answer}")
        return "sim" in answer
    except Exception as e:
        logging.error(f"Erro ao validar contexto com o Groq: {e}")
        return False

def acionar_agentes(user_id):
    logging.info("Iniciando o processo dos agentes do Crew AI.")
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.join(current_dir, 'src')
        main_py_path = os.path.join(src_dir, 'crew', 'main.py')
        if not os.path.isfile(main_py_path):
            logging.error(f"main.py não encontrado no caminho: {main_py_path}")
            return
        logging.debug(f"Executando o arquivo: {main_py_path} com user_id: {user_id}")
        subprocess.run(
            [sys.executable, main_py_path, user_id],
            check=True,
            cwd=src_dir
        )
        logging.info("Processo dos agentes concluído com sucesso.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Erro ao executar os agentes do Crew AI: {e}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")
    logging.info("Chain of agent execution:")

def adicionar_mensagem_ia(message, user_id, memory):
    memory.chat_memory.add_ai_message(message)
    salvar_memoria(user_id, memory.chat_memory.messages)
    logging.info(f"Mensagem da IA adicionada ao histórico: {message}")

def detectar_intencao_ai(usuario_resposta, contexto):
    prompt = (
        f"Abaixo está a conversa com um usuário. Baseado na última mensagem, determine se o usuário deseja receber "
        f"recomendações:\n\n"
        f"Contexto da conversa:\n{contexto}\n\n"
        f"Última mensagem do usuário:\n{usuario_resposta}\n\n"
        f"Responda apenas com 'sim' se a intenção do usuário for receber recomendações. Caso contrário, responda 'não'."
    )
    try:
        response = client.chat.completions.create(
            model="llama-3.2-90b-text-preview",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
            top_p=1,
            stream=False,
            stop=None,
        )
        resposta = response.choices[0].message.content.strip().lower()
        logging.info(f"Intenção detectada: {resposta}")
        return "sim" in resposta
    except Exception as e:
        logging.error(f"Erro ao detectar intenção com o Groq: {e}")
        return False

# Route for login
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    senha = data.get('senha')
    users = load_users()
    if email in users and users[email]['password'] == senha:
        user_id = users[email]['user_id']
        return jsonify({'sucesso': True, 'user_id': user_id})
    else:
        return jsonify({'sucesso': False, 'mensagem': 'Email ou senha incorretos.'})

# Route to load conversation
@app.route('/conversa', methods=['POST'])
def conversa():
    data = request.get_json()
    user_id = data.get('user_id')
    if not user_id:
        return jsonify({'messages': []})
    messages = carregar_memoria(user_id)
    if not messages:
        logging.info("Nenhuma mensagem encontrada. Gerando mensagem inicial.")
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        resposta_inicial = gerar_resposta_groq(memory.chat_memory.messages)
        memory.chat_memory.add_ai_message(resposta_inicial)
        salvar_memoria(user_id, memory.chat_memory.messages)
        messages = memory.chat_memory.messages
    messages_to_return = []
    for msg in messages:
        role = 'user' if isinstance(msg, HumanMessage) else 'bot'
        messages_to_return.append({'role': role, 'content': msg.content})
    return jsonify({'messages': messages_to_return})

# Route to send message to chatbot
@app.route('/mensagem', methods=['POST'])
def mensagem():
    data = request.get_json()
    mensagem_usuario = data.get('mensagem')
    user_id = data.get('user_id')
    if not user_id or not mensagem_usuario:
        return jsonify({'resposta': 'Dados inválidos.'})
    messages = carregar_memoria(user_id)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    memory.chat_memory.messages = messages
    memory.chat_memory.add_user_message(mensagem_usuario)
    armazenar_mensagem_no_vectorstore('user', mensagem_usuario, user_id)
    salvar_memoria(user_id, memory.chat_memory.messages)
    resposta_chatbot = gerar_resposta_groq(memory.chat_memory.messages)
    memory.chat_memory.add_ai_message(resposta_chatbot)
    salvar_memoria(user_id, memory.chat_memory.messages)
    contexto = "\n".join(
        f"{'Usuário' if isinstance(msg, HumanMessage) else 'Assistente'}: {msg.content}"
        for msg in memory.chat_memory.messages
    )
    if detectar_intencao_ai(mensagem_usuario, contexto):
        logging.info("Intenção de receber recomendações detectada pela IA.")
        if validar_contexto_suficiente(memory.chat_memory.messages):
            mensagem_ia = "Certo, processando suas recomendações."
            memory.chat_memory.add_ai_message(mensagem_ia)
            salvar_memoria(user_id, memory.chat_memory.messages)
            acionar_agentes(user_id)
            mostrar_oportunidades = True
        else:
            mensagem_ia = "Ainda preciso de mais algumas informações antes de enviar as recomendações. Vamos continuar nossa conversa."
            memory.chat_memory.add_ai_message(mensagem_ia)
            salvar_memoria(user_id, memory.chat_memory.messages)
            mostrar_oportunidades = False
        return jsonify({'resposta': resposta_chatbot + "\n" + mensagem_ia, 'mostrar_oportunidades': mostrar_oportunidades})
    else:
        mostrar_oportunidades = False
        return jsonify({'resposta': resposta_chatbot, 'mostrar_oportunidades': mostrar_oportunidades})

# Route to fetch opportunities
@app.route('/oportunidades', methods=['POST'])
def oportunidades():
    data = request.get_json()
    user_id = data.get('user_id')
    oportunidades = collection_oportunidades.find({'user_id': user_id})
    oportunidades_list = []
    for oportunidade in oportunidades:
        oportunidades_list.append({
            'titulo': oportunidade.get('titulo'),
            'descricao': oportunidade.get('descricao'),
            'link': oportunidade.get('link')
        })
    return jsonify({'oportunidades': oportunidades_list})

if __name__ == '__main__':
    app.run(debug=True)
