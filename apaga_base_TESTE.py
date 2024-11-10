import os
import logging
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import urllib.parse

# Carregar as variáveis de ambiente do arquivo .env
load_dotenv()

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
uri = f"mongodb+srv://{username}:{password}@{cluster_host}/?retryWrites=true&w=majority&appName=HackathonMeta&tls=true&tlsAllowInvalidCertificates=true"

# Criar o cliente MongoDB com ServerApi
client_mongo = MongoClient(uri, server_api=ServerApi('1'))

# Testar a conexão
try:
    client_mongo.admin.command('ping')
    logging.info("Conexão bem-sucedida com o MongoDB.")
except Exception as e:
    logging.error(f"Erro ao conectar ao MongoDB: {e}")
    raise

# Conexão com o banco de dados
db = client_mongo['DadosUsuários']

# Coleções a serem limpas
collections_to_clear = ['Contexto', 'HistoricoConversa']

# Função para limpar as coleções
def clear_collections():
    for collection_name in collections_to_clear:
        collection = db[collection_name]
        result = collection.delete_many({})
        logging.info(f"Coleção '{collection_name}' limpa. Documentos removidos: {result.deleted_count}")

if __name__ == '__main__':
    confirm = input("Você tem certeza que deseja limpar as coleções? Isso irá apagar todos os dados. (s/N): ")
    if confirm.lower() == 's':
        clear_collections()
        logging.info("Todas as coleções foram limpas com sucesso.")
    else:
        logging.info("Operação cancelada pelo usuário.")
