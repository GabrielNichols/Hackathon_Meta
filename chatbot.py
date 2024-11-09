import os
import json
import logging
from tqdm import tqdm
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage
from groq import Groq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Inicialize o cliente Groq com a chave da API
api_key = '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'  # Substitua pela sua chave de API
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
def carregar_memoria():
    logging.info("Carregando memória da conversa...")
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
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
def salvar_memoria(messages):
    logging.info("Salvando memória da conversa...")
    messages_data = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            msg_type = 'human'
        elif isinstance(msg, AIMessage):
            msg_type = 'ai'
        else:
            continue
        messages_data.append({'type': msg_type, 'content': msg.content})
    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(messages_data, f, ensure_ascii=False, indent=2)
    logging.info("Memória salva com sucesso.")

# Carrega a memória da conversa
messages = carregar_memoria()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
memory.chat_memory.messages = messages

# Função para gerar respostas com o modelo Groq
def gerar_resposta_groq(messages):
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
    message_id = f"{role}_{len(memory.chat_memory.messages)}"
    # Cria metadados para a mensagem
    metadata = {"role": role}
    # Adiciona a mensagem ao banco vetorial
    vectorstore.add_texts([content], metadatas=[metadata], ids=[message_id])
    logging.info(f"Mensagem armazenada no vectorstore com ID {message_id}.")

# Função para recuperar mensagens relevantes do banco vetorial
def recuperar_mensagens_relevantes(query, k=3):
    logging.info("Recuperando mensagens relevantes do vectorstore...")
    docs = vectorstore.similarity_search(query, k=k)
    mensagens = [doc.page_content for doc in docs]
    logging.info(f"{len(mensagens)} mensagens relevantes recuperadas.")
    return mensagens

# Função principal para iniciar a conversa usando memória de LangChain e integração com o modelo
def iniciar_conversa():
    logging.info("Iniciando a conversa...")
    with tqdm(total=100, desc="Progresso da Conversa") as pbar:
        while True:
            # Recupera mensagens relevantes do banco vetorial
            historico_formatado = ""
            mensagens_relevantes = recuperar_mensagens_relevantes(historico_formatado)
            contexto_relevante = "\n".join(mensagens_relevantes)

            # Gera a resposta do chatbot usando o modelo Groq
            resposta_chatbot = gerar_resposta_groq(memory.chat_memory.messages)

            # Exibe a resposta do chatbot e atualiza a memória com a resposta do chatbot
            print("\n\nChatbot:", resposta_chatbot)
            memory.chat_memory.add_ai_message(resposta_chatbot)

            # Armazena a resposta do chatbot no banco vetorial
            armazenar_mensagem_no_vectorstore('assistant', resposta_chatbot)

            # Salva a memória da conversa
            salvar_memoria(memory.chat_memory.messages)

            # Atualiza a barra de progresso
            pbar.update(10)

            # Verifica se a conversa foi concluída
            if "Obrigado pelas informações, vou analisar as oportunidades que se encaixam no seu perfil!" in resposta_chatbot:
                print("\nConversa concluída. Processaremos as informações coletadas.")
                logging.info("Conversa concluída.")
                break

            # Recebe a resposta do usuário e atualiza a memória
            usuario_resposta = input("\n\nVocê: ")
            memory.chat_memory.add_user_message(usuario_resposta)

            # Armazena a resposta do usuário no banco vetorial
            armazenar_mensagem_no_vectorstore('user', usuario_resposta)

            # Salva a memória da conversa
            salvar_memoria(memory.chat_memory.messages)

            # Atualiza a barra de progresso
            pbar.update(10)

    logging.info("Conversa encerrada.")

# Inicia o processo de conversa com as instruções do sistema
iniciar_conversa()

# Exibe o histórico final da conversa, coletado automaticamente por LangChain
print("\nHistórico Completo:")
for i, msg in enumerate(memory.chat_memory.messages):
    role = "Chatbot" if isinstance(msg, AIMessage) else "Você"
    print(f"{i + 1}. {role}: {msg.content}")
