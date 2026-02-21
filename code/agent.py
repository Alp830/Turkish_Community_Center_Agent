import os
from pathlib import Path

from google.adk.agents.llm_agent import Agent
from google.adk.tools.google_search_tool import GoogleSearchTool


def _load_env_if_present() -> None:
    """Load key/value pairs from .env if the file exists."""
    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


_load_env_if_present()
area = ""
root_agent = Agent(
    name="church_finder_agent",
    model="gemini-2.5-flash",
    description="Find churches in a user-provided area.",
    instruction=(
        "You are a helpful agent to help us find churches nearby so we can buy one and use as mosque"
        f""" 
            Once user promtps what location they are interested in finding mosques nearby, your job is to find ALL the churches in that location
            after user promtps, store users area in variable called 'area'
            use the search tool to search "Mosques in {area}"
            go through all the pages throughly, make sure to find EVERY church in palo alto you can, make a minmum of 25, but you are absolutely encoraged to go over if you find more mosques
            if the user asks for something bigger than a city (palo alto etc) make sure to boil down to churches in each city
            Make a list of ALL of the churches that show up in that area.
        """
    ),
    tools=[GoogleSearchTool(bypass_multi_tools_limit=True)],
)

# Alternative names that ADK might look for.
agent = root_agent
main_agent = root_agent
