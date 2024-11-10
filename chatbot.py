import os
import logging
from tqdm import tqdm
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
import sys  # Ensure sys is imported for path operations

# Carregar as variáveis de ambiente do arquivo .env
load_dotenv()

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Inicialize o cliente Groq com a chave da API
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("A chave de API do Groq não foi encontrada. Verifique se está definida no arquivo .env.")
client = Groq(api_key=api_key)

# Inicializar a chave de API do Cohere
cohere_api_key = os.getenv("COHERE_API_KEY")
if not cohere_api_key:
    raise ValueError("A chave de API do Cohere não foi encontrada. Verifique se está definida no arquivo .env.")

# Criar a classe de embeddings com a API da Cohere
class CohereEmbeddings(Embeddings):
    def __init__(self, api_key, model="embed-english-light-v3.0", input_type="search_query", embedding_types=["float"]):
        self.client = cohere.ClientV2(api_key)
        self.model = model
        self.input_type = input_type
        self.embedding_types = embedding_types
    
    def embed_documents(self, texts):
        response = self.client.embed(
            texts=texts,
            model=self.model,
            input_type=self.input_type,
            embedding_types=self.embedding_types,
        )
        return response.embeddings.float  # Alterado aqui
    
    def embed_query(self, text):
        response = self.client.embed(
            texts=[text],
            model=self.model,
            input_type=self.input_type,
            embedding_types=self.embedding_types,
        )
        return response.embeddings.float[0]  # Alterado aqui

# Inicialize o modelo de embeddings com o Cohere
embedding_model = CohereEmbeddings(
    api_key=cohere_api_key,
    model="embed-multilingual-light-v3.0",  # Escolha o modelo apropriado
    input_type="search_query",         # Ajuste conforme necessário
    embedding_types=["float"]          # Tipos de embeddings desejados
)

# Carregar as credenciais do MongoDB
MONGODB_USERNAME = os.getenv('MONGODB_USERNAME')
MONGODB_PASSWORD = os.getenv('MONGODB_PASSWORD')

if not MONGODB_USERNAME or not MONGODB_PASSWORD:
    raise ValueError("Nome de usuário ou senha do MongoDB não encontrados no arquivo .env.")

# Codificar o nome de usuário e a senha
username = urllib.parse.quote_plus(MONGODB_USERNAME)
password = urllib.parse.quote_plus(MONGODB_PASSWORD)

# Nome do host do cluster (substitua pelo seu)
cluster_host = 'hackathonmeta.pvjrb.mongodb.net'  # Certifique-se de que este é o nome correto do seu cluster

# Montar a string de conexão
uri = f"mongodb+srv://{username}:{password}@{cluster_host}/?retryWrites=true&w=majority&appName=HackathonMeta&tls=true"

# Criar o cliente MongoDB com ServerApi
client_mongo = MongoClient(uri, server_api=ServerApi('1'))

# Testar a conexão
try:
    client_mongo.admin.command('ping')
    logging.info("Conexão bem-sucedida com o MongoDB.")
except Exception as e:
    logging.error(f"Erro ao conectar ao MongoDB: {e}")
    raise

# Conexão com o banco de dados e coleções
db = client_mongo['DadosUsuários']
collection_contexto = db['Contexto']
collection_historico = db['HistoricoConversa']

# Inicializa o MongoDB Atlas Vector Search
vectorstore = MongoDBAtlasVectorSearch(
    collection=collection_contexto,
    embedding=embedding_model,
    index_name='contexto',
    text_key='content',
    embedding_key='embedding'
)

# Defina o USER_ID (obtenha do sistema de autenticação)
USER_ID = 'user123'  # Substitua pelo identificador real do usuário

# Refine the system_prompt to be more precise
system_prompt = (
    "Você é um assistente especializado em ajudar usuários a encontrar oportunidades de desenvolvimento profissional. "
    "Suas respostas devem ser claras, concisas e diretas ao ponto, mantendo uma abordagem amigável e informativa. "
    "Conduza a conversa de forma estruturada para coletar as seguintes informações essenciais do usuário: "
    "1. Nível de escolaridade atual e desejada. "
    "2. Área de trabalho atual e nível de satisfação com o emprego atual. "
    "3. Objetivos profissionais específicos a curto e longo prazo. "
    "4. Cursos, treinamentos ou certificações que o usuário deseja realizar, incluindo áreas de interesse. "
    "5. Preferência por oportunidades presenciais, online ou híbridas. "
    "6. Limitações de tempo, financeiras ou outras que possam impactar a participação em oportunidades. "
    "Caso o usuário solicite recomendações antes de fornecer todas as informações necessárias, informe que precisa de mais detalhes para fornecer sugestões adequadas e continue a coleta das informações. "
    "Se todas as informações forem coletadas, pergunte ao usuário se deseja receber as recomendações agora. "
    "Lembre-se: Você não gera as recomendações diretamente, mas coleta informações precisas para que os agentes do Crew AI possam processá-las eficientemente."
)

# Função para carregar a memória da conversa
def carregar_memoria():
    logging.info("Carregando memória da conversa do MongoDB...")
    conversa = collection_historico.find_one({'user_id': USER_ID})
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

# Função para salvar a memória da conversa
def salvar_memoria(messages):
    logging.info("Salvando memória da conversa no MongoDB...")
    # Converter as mensagens para um formato serializável
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
    # Criar o documento a ser inserido
    conversa = {
        'user_id': USER_ID,
        'messages': messages_data,
        'last_updated': datetime.datetime.utcnow()
    }
    # Inserir ou atualizar a conversa no MongoDB
    collection_historico.update_one(
        {'user_id': USER_ID},
        {'$set': conversa},
        upsert=True
    )
    logging.info("Memória da conversa salva no MongoDB.")

# Carrega a memória da conversa
messages = carregar_memoria()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
memory.chat_memory.messages = messages

def gerar_resposta_groq(messages):
    logging.info("Gerando resposta do modelo Groq...")
    # Construindo as mensagens para o modelo
    model_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        if isinstance(msg, HumanMessage):
            model_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            model_messages.append({"role": "assistant", "content": msg.content})

    # Chamando a API do Groq
    try:
        completion = client.chat.completions.create(
            model="llama-3.2-90b-text-preview",
            messages=model_messages,
            temperature=0.7,
            max_tokens=820,
            top_p=1,
            stream=False,
            stop=None,
        )
        response = completion.choices[0].message.content.strip()
        logging.info("Resposta gerada com sucesso.")
        return response
    except Exception as e:
        logging.error(f"Erro ao gerar resposta com o Groq: {e}")
        return "Houve um erro ao processar sua solicitação."


# Função para armazenar mensagens no banco vetorial (apenas do usuário)
def armazenar_mensagem_no_vectorstore(role, content):
    if role == 'user':
        # Cria metadados para a mensagem, incluindo o user_id
        metadata = {"role": role, "user_id": USER_ID}
        # Adiciona a mensagem ao vector store
        vectorstore.add_texts([content], metadatas=[metadata])
        logging.info("Mensagem do usuário armazenada no vectorstore.")
    else:
        logging.info("Mensagem do assistente não armazenada no vectorstore.")

# Função para validar se o contexto é suficiente
def validar_contexto_suficiente(messages):
    logging.info("Validando se o contexto é suficiente para gerar recomendações.")
    # Cria o prompt de validação
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

    # Formata a conversa
    conversation_text = ""
    for msg in messages:
        role = "Assistente" if isinstance(msg, AIMessage) else "Usuário"
        conversation_text += f"{role}: {msg.content}\n"

    # Insere a conversa no prompt
    formatted_prompt = validation_prompt.format(conversation=conversation_text)

    # Chama o modelo de IA do Groq
    try:
        completion = client.chat.completions.create(
            model="llama-3.2-90b-text-preview",
            messages=[{"role": "system", "content": formatted_prompt}],
            temperature=0.0,
            max_tokens=10,
            top_p=1,
            stream=False,
            stop=None,
        )
        answer = completion.choices[0].message.content.strip().lower()
        logging.info(f"Resposta da validação: {answer}")
        return "sim" in answer
    except Exception as e:
        logging.error(f"Erro ao validar contexto com o Groq: {e}")
        return False

# Função para acionar os agentes do Crew AI
def acionar_agentes():
    """
    Aciona os agentes do CrewAI e mostra as oportunidades.
    """
    logging.info("Iniciando o processo dos agentes do Crew AI.")
    try:
        # Construct the absolute path to main.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.join(current_dir, 'src')  # Set to 'src' directory
        main_py_path = os.path.join(src_dir, 'crew', 'main.py')        
        # Verify if main.py exists at the constructed path
        if not os.path.isfile(main_py_path):
            logging.error(f"main.py não encontrado no caminho: {main_py_path}")
            return
        
        # Log the path being used
        logging.debug(f"Executando o arquivo: {main_py_path} com user_id: {USER_ID}")
        
        # Execute main.py with the correct path and set the working directory to 'src'
        subprocess.run(
            [sys.executable, main_py_path, USER_ID],
            check=True,
            cwd=src_dir  # Set working directory to 'src'
        )
        logging.info("Processo dos agentes concluído com sucesso.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Erro ao executar os agentes do Crew AI: {e}")
    except Exception as e:
        logging.error(f"Erro inesperado: {e}")

    # Log the execution chain
    logging.info("Chain of agent execution:")
    # Assuming that 'main.py' logs the agents' execution, otherwise implement additional logging here

def adicionar_mensagem_ia(message):
    """
    Adiciona uma mensagem da IA ao histórico de mensagens e salva no banco.
    """
    memory.chat_memory.add_ai_message(message)
    salvar_memoria(memory.chat_memory.messages)
    logging.info(f"Mensagem da IA adicionada ao histórico: {message}")

def detectar_intencao_ai(usuario_resposta, contexto):
    """
    Identifica a intenção do usuário usando o modelo de IA.
    """
    prompt = (
        f"Abaixo está a conversa com um usuário. Baseado na última mensagem, determine se o usuário deseja receber "
        f"recomendações:\n\n"
        f"Contexto da conversa:\n{contexto}\n\n"
        f"Última mensagem do usuário:\n{usuario_resposta}\n\n"
        f"Responda apenas com 'sim' se a intenção do usuário for receber recomendações. Caso contrário, responda 'não'."
    )
    
    # Chamando o modelo de IA do Groq
    response = client.chat.completions.create(
        model="llama-3.2-90b-text-preview",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.0,  # Sem variação na resposta
        max_tokens=10,  # Apenas 'sim' ou 'não'
        top_p=1,
        stream=False,
        stop=None,
    )
    
    # Retorna True se a resposta for 'sim', False caso contrário
    resposta = response.choices[0].message.content.strip().lower()
    logging.info(f"Intenção detectada: {resposta}")
    return "sim" in resposta


def iniciar_conversa():
    logging.info("Iniciando a conversa...")
    with tqdm(total=100, desc="Progresso da Conversa") as pbar:
        while True:
            # Gera a resposta do chatbot usando o modelo Groq
            resposta_chatbot = gerar_resposta_groq(memory.chat_memory.messages)
            
            # Exibe e salva a resposta do chatbot
            print("\n\nChatbot:", resposta_chatbot)
            adicionar_mensagem_ia(resposta_chatbot)
            pbar.update(10)

            # Recebe a resposta do usuário e atualiza a memória
            usuario_resposta = input("\n\nVocê: ")
            memory.chat_memory.add_user_message(usuario_resposta)
            armazenar_mensagem_no_vectorstore('user', usuario_resposta)
            salvar_memoria(memory.chat_memory.messages)
            pbar.update(10)

            # Valida intenção com IA
            contexto = "\n".join(
                f"{'Usuário' if isinstance(msg, HumanMessage) else 'Assistente'}: {msg.content}"
                for msg in memory.chat_memory.messages
            )
            
            if detectar_intencao_ai(usuario_resposta, contexto):
                logging.info("Intenção de receber recomendações detectada pela IA.")
                if validar_contexto_suficiente(memory.chat_memory.messages):
                    mensagem_ia = "Certo, processando suas recomendações."
                    print("\nChatbot:", mensagem_ia)
                    adicionar_mensagem_ia(mensagem_ia)
                    acionar_agentes()
                    break
                else:
                    mensagem_ia = "Ainda preciso de mais algumas informações antes de enviar as recomendações. Vamos continuar nossa conversa."
                    print("\nChatbot:", mensagem_ia)
                    adicionar_mensagem_ia(mensagem_ia)

            # Verifica se a conversa foi concluída pelo assistente
            if "Obrigado pelas informações, vou analisar as oportunidades que se encaixam no seu perfil!" in resposta_chatbot:
                mensagem_ia = "Conversa concluída. Processaremos as informações coletadas."
                print("\nChatbot:", mensagem_ia)
                adicionar_mensagem_ia(mensagem_ia)
                acionar_agentes()
                break

    logging.info("Conversa encerrada.")

# Inicia o processo de conversa
iniciar_conversa()

# Exibe o histórico final da conversa
print("\nHistórico Completo:")
for i, msg in enumerate(memory.chat_memory.messages):
    role = "Chatbot" if isinstance(msg, AIMessage) else "Você"
    print(f"{i + 1}. {role}: {msg.content}")
