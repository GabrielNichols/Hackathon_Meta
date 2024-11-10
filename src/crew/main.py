#!/usr/bin/env python
import sys
from crew import OportunityFinderCrew
import logging  # Add import for logging if not present
import os  # Ensure os is imported for path operations

def run():
    """
    Executa o crew.
    """
    # Add the 'src' directory to sys.path to resolve imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(current_dir, '..'))
    if src_dir not in sys.path:
        sys.path.append(src_dir)
        logging.debug(f"Adicionado ao sys.path: {src_dir}")
    
    user_id = sys.argv[1] if len(sys.argv) > 1 else 'user123'
    crew_instance = OportunityFinderCrew(user_id=user_id)

    logging.info("Iniciando a execução das tarefas do crew.")
    # Executa todas as tarefas do crew
    crew_instance.crew().kickoff()
    logging.info("Execução das tarefas do crew concluída.")

    # Mostra as oportunidades salvas no MongoDB
    mostrar_oportunidades(user_id)


def mostrar_oportunidades(user_id):
    """
    Exibe as oportunidades salvas na coleção 'Oportunidades'.
    """
    # Conecta ao banco de dados MongoDB
    crew_instance = OportunityFinderCrew(user_id=user_id)
    db = crew_instance.app.db  # Updated to access MongoDB instance
    collection_opportunities = db['Oportunidades']

    # Recupera as oportunidades do usuário
    oportunidades = collection_opportunities.find_one({'user_id': user_id})

    if not oportunidades:
        print("\nNenhuma oportunidade encontrada no momento.")
        return

    print("\n\n--- Oportunidades Encontradas ---")
    if 'trabalho' in oportunidades:
        print("\n- **Oportunidades de Trabalho**:")
        for trabalho in oportunidades['trabalho']:
            print(f"  - {trabalho}")

    if 'educacao' in oportunidades:
        print("\n- **Oportunidades de Educação**:")
        for curso in oportunidades['educacao']:
            print(f"  - {curso}")

    if 'evento' in oportunidades:
        print("\n- **Oportunidades de Eventos**:")
        for evento in oportunidades['evento']:
            print(f"  - {evento}")

if __name__ == "__main__":
    run()
