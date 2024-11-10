import logging
from src.crew.crew import OportunityFinderCrew
import os
from dotenv import load_dotenv

# Configurar variáveis de ambiente
load_dotenv()

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Definir USER_ID para teste
USER_ID = "user123"  # Substitua pelo identificador real do usuário


def main():
    """
    Testa o fluxo de obtenção de oportunidades do Crew AI.
    """
    logging.info("Iniciando o teste de oportunidades...")

    # Verificar se a chave da API está configurada
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        logging.error("COHERE_API_KEY não está definida no ambiente.")
        return

    logging.info("Instanciando OportunityFinderCrew...")
    crew_instance = OportunityFinderCrew(user_id=USER_ID)
    logging.debug(f"crew_instance: {crew_instance}")

    # Inicializar o Crew
    crew = crew_instance.crew()

    # Executar o Crew
    logging.info("Executando o Crew para buscar oportunidades...")
    crew.kickoff()
    logging.info("Crew executado com sucesso.")

    # Simular resultados das tarefas
    logging.info("Buscando oportunidades do MongoDB...")
    
    # Obter oportunidades do MongoDB
    saved_opportunities = crew_instance.get_saved_opportunities(user_id=USER_ID)
    job_opportunities = saved_opportunities.get("job_opportunities", [])
    event_opportunities = saved_opportunities.get("event_opportunities", [])
    course_opportunities = saved_opportunities.get("course_opportunities", [])

    logging.info("Processando e salvando oportunidades...")
    crew_instance.process_and_save_opportunities(
        user_id=USER_ID,
        job_opportunities=job_opportunities,
        event_opportunities=event_opportunities,
        course_opportunities=course_opportunities
    )
    logging.info("Oportunidades processadas e salvas.")

    # Exibir as oportunidades salvas
    logging.info("Exibindo as oportunidades encontradas...")
    crew_instance.display_opportunities(user_id=USER_ID)
    logging.info("Exibição concluída.")


if __name__ == "__main__":
    main()
