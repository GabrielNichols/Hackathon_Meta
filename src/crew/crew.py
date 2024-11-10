import logging
from crewai_tools import (
    SerperDevTool,
    ScrapeWebsiteTool,
    SerplyJobSearchTool,
    SerplyNewsSearchTool,
    SerplyWebSearchTool,
)
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
import os
from pymongo import MongoClient
from dotenv import load_dotenv
import urllib.parse
import datetime

from litellm import completion  # Use `chat_completion` instead of `completion`
from crewai.llm import LLM  # Import LLM from crewai.llm

# Inicialize o cliente Groq com a chave da API
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("A chave de API do Groq não foi encontrada. Verifique se está definida no arquivo .env.")

# Defina o modelo para o LiteLLM (Groq Llama)
MODEL_NAME = "groq/llama-3.2-90b-text-preview"

# Função para gerar respostas usando o LiteLLM com o modelo da Groq
def generate_response(messages):
    response = completion(
        model=MODEL_NAME,
        messages=messages,
        api_key=api_key,  # Passa a chave da API diretamente aqui
    )
    return response

class MongoDBApp:
    def __init__(self):
        logging.info("Initializing MongoDBApp...")

        # Configure MongoDB connection
        MONGODB_USERNAME = os.getenv('MONGODB_USERNAME')
        MONGODB_PASSWORD = os.getenv('MONGODB_PASSWORD')

        if not MONGODB_USERNAME or not MONGODB_PASSWORD:
            raise ValueError("MongoDB username or password not found in .env file.")

        # Encode credentials
        username = urllib.parse.quote_plus(MONGODB_USERNAME)
        password = urllib.parse.quote_plus(MONGODB_PASSWORD)

        # Configure cluster URI
        cluster_host = 'hackathonmeta.pvjrb.mongodb.net'
        uri = f"mongodb+srv://{username}:{password}@{cluster_host}/?retryWrites=true&w=majority"

        # Connect to MongoDB
        self.client = MongoClient(uri)
        self.db = self.client['DadosUsuários']
        logging.debug("MongoDB connection established.")

    def get_context_collection(self):
        return self.db['Contexto']

    def get_opportunities_collection(self):
        return self.db['Oportunidades']

@CrewBase
class OportunityFinderCrew:
    """Crew for finding and processing opportunities."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self, user_id):
        logging.info(f"Initializing OportunityFinderCrew for user_id: {user_id}")
        self.user_id = user_id
        self.app = MongoDBApp()
        logging.debug("OportunityFinderCrew initialized.")

    @agent
    def user_context_analyzer(self) -> Agent:
        """Agent to analyze user context."""

        def analyze_context():
            logging.info("Agent 'user_context_analyzer' iniciado.")
            collection_context = self.app.get_context_collection()
            user_context = collection_context.find_one({"user_id": self.user_id})
            logging.debug(f"User context retrieved: {user_context}")
            logging.info("Agent 'user_context_analyzer' concluído.")
            return user_context or {}

        return Agent(
            role="Coletor de Contexto do Usuário",
            goal="Coletar e analisar o contexto de informações do usuário a partir do banco de dados vetorizado.",
            backstory="Você é um analista de dados responsável por coletar informações detalhadas sobre a situação socioeconômica, interesses e objetivos do usuário. Seu objetivo é entender o contexto vetorizado que será usado pelos agentes para sugerir oportunidades personalizadas ao usuário.",
            verbose=True,
            analyze_context=analyze_context
        )

    @agent
    def job_opportunity_finder(self) -> Agent:
        """Agent to find job opportunities."""
        
        def search_jobs(context):
            logging.info("Agent 'job_opportunity_finder' iniciado.")
            keywords = context.get("job_preferences", {}).get("keywords", [])
            location = context.get("location", "global")
            max_results = context.get("job_preferences", {}).get("max_results", 10)
            logging.debug(f"Buscar empregos com keywords: {keywords}, location: {location}, max_results: {max_results}")
            logging.info("Agent 'job_opportunity_finder' concluído.")
            return {
                "keywords": keywords,
                "location": location,
                "max_results": max_results
            }

        return Agent(
            role="Encontrador de Oportunidades de Emprego",
            goal="Identificar e listar oportunidades de emprego que correspondam ao perfil e interesses do usuário.",
            backstory="Você é especialista em identificar oportunidades de emprego que correspondem ao perfil e interesses do usuário. Utilize o contexto fornecido para buscar oportunidades relevantes que se alinham aos objetivos do usuário.",
            tools=[
                SerplyJobSearchTool(
                    keywords=search_jobs,
                    location=search_jobs,
                    max_results=search_jobs
                )
            ],
            verbose=True,
        )

    @agent
    def event_opportunity_finder(self) -> Agent:
        """Agent to find event opportunities."""
        
        def search_events(context):
            logging.info("Agent 'event_opportunity_finder' iniciado.")
            event_types = context.get("event_preferences", {}).get("types", [])
            location = context.get("location", "global")
            limit = context.get("event_preferences", {}).get("limit", 10)
            logging.debug(f"Buscar eventos com types: {event_types}, location: {location}, limit: {limit}")
            logging.info("Agent 'event_opportunity_finder' concluído.")
            return {
                "event_types": event_types,
                "location": location,
                "limit": limit
            }

        return Agent(
            role="Encontrador de Oportunidades de Eventos",
            goal="Descobrir eventos que contribuam para o desenvolvimento pessoal e profissional do usuário.",
            backstory="Você identifica eventos que podem beneficiar o crescimento pessoal e profissional do usuário, utilizando o contexto coletado para encontrar eventos que correspondem aos interesses e objetivos do usuário.",
            tools=[
                SerplyNewsSearchTool(
                    event_types=search_events,
                    location=search_events,
                    limit=search_events
                )
            ],
            verbose=True,
        )

    @agent
    def course_opportunity_finder(self) -> Agent:
        """Agent to find course opportunities."""
        
        def search_courses(context):
            logging.info("Agent 'course_opportunity_finder' iniciado.")
            topics = context.get("course_preferences", {}).get("topics", [])
            modality = context.get("course_preferences", {}).get("modality", "online")
            limit = context.get("course_preferences", {}).get("limit", 10)
            logging.debug(f"Buscar cursos com topics: {topics}, modality: {modality}, limit: {limit}")
            logging.info("Agent 'course_opportunity_finder' concluído.")
            return {
                "topics": topics,
                "modality": modality,
                "limit": limit
            }

        return Agent(
            role="Encontrador de Oportunidades de Cursos",
            goal="Encontrar cursos que atendam às necessidades educacionais e interesses do usuário.",
            backstory="Você busca cursos que correspondem ao perfil educacional e aos interesses do usuário, ajudando-o a adquirir novas habilidades e conhecimentos conforme seus objetivos.",
            tools=[
                SerplyWebSearchTool(
                    topics=search_courses,
                    modality=search_courses,
                    limit=search_courses
                )
            ],
            verbose=True,
        )

    @agent
    def professional_development_finder(self) -> Agent:
        """Agent to find professional development opportunities."""
        
        def search_professional_development(context):
            logging.info("Agent 'professional_development_finder' iniciado.")
            development_types = context.get("professional_development_preferences", {}).get("types", [])
            location = context.get("location", "global")
            limit = context.get("professional_development_preferences", {}).get("limit", 10)
            logging.debug(f"Buscar desenvolvimento profissional com types: {development_types}, location: {location}, limit: {limit}")
            logging.info("Agent 'professional_development_finder' concluído.")
            return {
                "development_types": development_types,
                "location": location,
                "limit": limit
            }

        return Agent(
            role="Encontrador de Desenvolvimento Profissional",
            goal="Identificar oportunidades de desenvolvimento profissional alinhadas com os objetivos do usuário.",
            backstory="Você procura por oportunidades que podem impulsionar a carreira do usuário, como programas de mentoria, workshops e outras atividades que promovem o crescimento profissional.",
            tools=[
                SerplyWebSearchTool(
                    development_types=search_professional_development,
                    location=search_professional_development,
                    limit=search_professional_development
                )
            ],
            verbose=True,
        )

    @task
    def analyze_user_context_task(self) -> Task:
        """Task to analyze user context."""
        return Task(
            config=self.tasks_config["collect_user_context_task"],
            agent=self.user_context_analyzer(),
        )

    @task
    def find_job_opportunities_task(self) -> Task:
        """Task to find job opportunities."""
        return Task(
            config=self.tasks_config["find_job_opportunities_task"],
            agent=self.job_opportunity_finder(),
            inputs=["analyze_user_context_task"],  # Takes user context as input
        )

    @task
    def find_event_opportunities_task(self) -> Task:
        """Task to find event opportunities."""
        return Task(
            config=self.tasks_config["find_event_opportunities_task"],
            agent=self.event_opportunity_finder(),
            inputs=["analyze_user_context_task"],  # Takes user context as input
        )

    @task
    def find_course_opportunities_task(self) -> Task:
        """Task to find course opportunities."""
        return Task(
            config=self.tasks_config["find_course_opportunities_task"],
            agent=self.course_opportunity_finder(),
            inputs=["analyze_user_context_task"],  # Takes user context as input
        )

    @task
    def find_professional_development_task(self) -> Task:
        """Task to find professional development opportunities."""
        return Task(
            config=self.tasks_config["find_professional_development_task"],
            agent=self.professional_development_finder(),
            inputs=["analyze_user_context_task"],  # Takes user context as input
        )

    @crew
    def crew(self) -> Crew:
        # Create an instance of LLM with the correct model and API key
        llm_instance = LLM(
            model=MODEL_NAME,
            api_key=api_key,
        )
        return Crew(
            agents=[
                self.user_context_analyzer(),
                self.job_opportunity_finder(),
                self.event_opportunity_finder(),
                self.course_opportunity_finder(),
                self.professional_development_finder(),
            ],
            tasks=[
                self.analyze_user_context_task(),
                self.find_job_opportunities_task(),
                self.find_event_opportunities_task(),
                self.find_course_opportunities_task(),
                self.find_professional_development_task(),
            ],
            # Use the llm_instance for manager_llm
            manager_llm=llm_instance,
            process=Process.hierarchical,
            respect_context_window=True,
            memory=True,
            planning=True,
        )

