from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent

from hermes.tools.perception_tools import geo_perception_tool
from hermes.tools.restoration_tools import GeoImageRestorationTool
from hermes.tools.damage_tools import GeoDamageAnalysisTool
from hermes.tools.reasoning_tools import GeoReasoningTool


@CrewBase
class Hermes():
    """Hermes crew"""

    agents: list[BaseAgent]
    tasks: list[Task]

    @agent
    def geo_perception_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["geo_perception_agent"],  # type: ignore[index]
            tools=[geo_perception_tool],
            verbose=True,
        )

    @agent
    def geo_image_restoration_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["geo_image_restoration_agent"],  # type: ignore[index]
            tools=[GeoImageRestorationTool()],
            verbose=True,
        )

    @agent
    def geo_damage_analysis_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["geo_damage_analysis_agent"],  # type: ignore[index]
            tools=[GeoDamageAnalysisTool()],
            verbose=True,
        )

    @agent
    def geo_reasoning_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["geo_reasoning_agent"],  # type: ignore[index]
            tools=[GeoReasoningTool()],
            verbose=True,
        )

    @task
    def geo_perception_task(self) -> Task:
        return Task(
            config=self.tasks_config["geo_perception_task"],  # type: ignore[index]
        )

    @task
    def geo_image_restoration_task(self) -> Task:
        return Task(
            config=self.tasks_config["geo_image_restoration_task"],  # type: ignore[index]
        )

    @task
    def geo_damage_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["geo_damage_analysis_task"],  # type: ignore[index]
        )

    @task
    def geo_reasoning_task(self) -> Task:
        return Task(
            config=self.tasks_config["geo_reasoning_task"],  # type: ignore[index]
            context=[
                self.geo_perception_task(),
                self.geo_image_restoration_task(),
                self.geo_damage_analysis_task(),
            ],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the HERMES crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )