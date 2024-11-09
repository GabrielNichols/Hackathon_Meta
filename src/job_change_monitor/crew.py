from crewai_tools import (
    SerperDevTool,
    ScrapeWebsiteTool,
    SerplyJobSearchTool,
    SerplyNewsSearchTool,
    SerplyWebSearchTool,
)

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task


@CrewBase
class JobChangeMonitorCrew:
    """JobChangeMonitor crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def user_context_collector(self) -> Agent:
        return Agent(
            config=self.agents_config["user_context_collector"],
            tools=[],  # Adicione ferramentas relevantes para interação via chatbot, se necessário
            allow_delegation=False,
            verbose=True,
        )

    @agent
    def job_opportunity_finder(self) -> Agent:
        return Agent(
            config=self.agents_config["job_opportunity_finder"],
            tools=[SerperDevTool(), SerplyJobSearchTool(), ScrapeWebsiteTool()],
            allow_delegation=False,
            verbose=True,
        )

    @agent
    def event_opportunity_finder(self) -> Agent:
        return Agent(
            config=self.agents_config["event_opportunity_finder"],
            tools=[SerperDevTool(), SerplyNewsSearchTool(), ScrapeWebsiteTool()],
            allow_delegation=False,
            verbose=True,
        )

    @agent
    def course_opportunity_finder(self) -> Agent:
        return Agent(
            config=self.agents_config["course_opportunity_finder"],
            tools=[SerperDevTool(), SerplyWebSearchTool(), ScrapeWebsiteTool()],
            allow_delegation=False,
            verbose=True,
        )

    @agent
    def professional_development_finder(self) -> Agent:
        return Agent(
            config=self.agents_config["professional_development_finder"],
            tools=[SerperDevTool(), SerplyWebSearchTool(), ScrapeWebsiteTool()],
            allow_delegation=False,
            verbose=True,
        )

    @task
    def collect_user_context_task(self) -> Task:
        return Task(
            config=self.tasks_config["collect_user_context_task"],
            agent=self.user_context_collector(),
            output_file="user_context.json",
        )

    @task
    def find_job_opportunities_task(self) -> Task:
        return Task(
            config=self.tasks_config["find_job_opportunities_task"],
            agent=self.job_opportunity_finder(),
            inputs=["user_context.json"],  # Utiliza o contexto do usuário como entrada
        )

    @task
    def find_event_opportunities_task(self) -> Task:
        return Task(
            config=self.tasks_config["find_event_opportunities_task"],
            agent=self.event_opportunity_finder(),
            inputs=["user_context.json"],
        )

    @task
    def find_course_opportunities_task(self) -> Task:
        return Task(
            config=self.tasks_config["find_course_opportunities_task"],
            agent=self.course_opportunity_finder(),
            inputs=["user_context.json"],
        )

    @task
    def find_professional_development_task(self) -> Task:
        return Task(
            config=self.tasks_config["find_professional_development_task"],
            agent=self.professional_development_finder(),
            inputs=["user_context.json"],
        )

    @crew
    def crew(self) -> Crew:
        """Cria o crew JobChangeMonitor"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
