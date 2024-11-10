import os
import json
import logging
from flask import Flask, request, jsonify
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage
from groq import Groq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Inicialize o cliente Groq com a chave da API
api_key = 'SUA_CHAVE_API_GROQ'  # Substitua pela sua chave de API
client = Groq(api_key=api_key)

# Diretórios para salvar a memória da conversa e o banco vetorial
MEMORY_FILE = 'conversa_memoria.json'
VECTORSTORE_DIR = 'vectorstore_data'

# Inicialize o modelo de embeddings e o banco vetorial com persistência
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Verifica se o diretório do vectorstore existe, se não, cria
if not os.path.exists(VECTORSTORE_DIR):
    os.makedirs(VECTORSTORE_DIR)

# Inicializa o Chroma com persistência
vectorstore = Chroma(embedding_function=embedding_model, persist_directory=VECTORSTORE_DIR)

# Prompt do sistema para guiar a conversa
system_prompt = (
    "Você é um assistente que ajuda usuários a encontrar oportunidades de desenvolvimento profissional. "
    "Conduza a conversa para coletar as seguintes informações: "
    "1. Nível de escolaridade do usuário. "
    "2. Área de trabalho atual e satisfação com o trabalho. "
    "3. Objetivo profissional específico. "
    "4. Cursos ou treinamentos desejados e áreas de interesse. "
    "5. Preferência por oportunidades presenciais ou virtuais. "
    "6. Limitações de tempo ou recursos que possam afetar o aproveitamento das oportunidades. "
    "Conduza a conversa de forma direta e evite sugestões detalhadas até que todas as informações sejam coletadas. "
    "Forneça respostas objetivas e peça mais detalhes apenas se necessário. "
    "Finalize a conversa dizendo: 'Obrigado pelas informações, vou analisar as oportunidades que se encaixam no seu perfil!' quando todos os dados tiverem sido coletados."
)

# Função para carregar a memória da conversa
def carregar_memoria(session_id):
    logging.info(f"Carregando memória da conversa para a sessão {session_id}...")
    filename = f'conversa_memoria_{session_id}.json'
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
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
    filename = f'conversa_memoria_{session_id}.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(messages_data, f, ensure_ascii=False, indent=2)
    logging.info("Memória salva com sucesso.")

# Função para gerar respostas com o modelo Groq
def gerar_resposta_groq(memory):
    logging.info("Gerando resposta do modelo Groq...")
    # Construindo as mensagens para o modelo
    model_messages = [{"role": "system", "content": system_prompt}]
    for msg in memory.chat_memory.messages:
        if isinstance(msg, HumanMessage):
            model_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            model_messages.append({"role": "assistant", "content": msg.content})
    # Chamando a API do modelo
    response = client.chat.completions.create(
        model="llama-3.2-90b-text-preview",
        messages=model_messages,
        temperature=0.3,
        max_tokens=150,
        top_p=1,
        stream=False,
        stop=None,
    )
    resposta = response.choices[0].message.content.strip()
    logging.info("Resposta gerada com sucesso.")
    return resposta

# Função para armazenar mensagens no banco vetorial
def armazenar_mensagem_no_vectorstore(role, content):
    # Cria um ID único para a mensagem
    message_id = f"{role}_{content[:10]}"  # Usando o início do conteúdo para identificação
    # Cria metadados para a mensagem
    metadata = {"role": role}
    # Adiciona a mensagem ao banco vetorial
    vectorstore.add_texts([content], metadatas=[metadata], ids=[message_id])
    logging.info(f"Mensagem armazenada no vectorstore com ID {message_id}.")

# Iniciando o aplicativo Flask
app = Flask(__name__)

# Rota para processar as mensagens do usuário
@app.route('/mensagem', methods=['POST'])
def processar_mensagem():
    dados = request.json
    mensagem_usuario = dados.get('mensagem', '')
    session_id = dados.get('session_id', 'default_session')

    # Carrega a memória da conversa para a sessão atual
    messages = carregar_memoria(session_id)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    memory.chat_memory.messages = messages

    # Atualiza a memória com a mensagem do usuário
    memory.chat_memory.add_user_message(mensagem_usuario)
    armazenar_mensagem_no_vectorstore('user', mensagem_usuario)
    salvar_memoria(memory.chat_memory.messages, session_id)

    # Gera a resposta do chatbot
    resposta_chatbot = gerar_resposta_groq(memory)

    # Atualiza a memória com a resposta do chatbot
    memory.chat_memory.add_ai_message(resposta_chatbot)
    armazenar_mensagem_no_vectorstore('assistant', resposta_chatbot)
    salvar_memoria(memory.chat_memory.messages, session_id)

    # Retorna a resposta para o front-end
    return jsonify({"resposta": resposta_chatbot})

if __name__ == '__main__':
    app.run(debug=True)
