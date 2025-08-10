import asyncio
from typing import Annotated
import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import TextContent, ImageContent, INVALID_PARAMS, INTERNAL_ERROR
from pydantic import BaseModel, Field, AnyUrl
from typing import Any, Dict, List, Optional
from ddgs import DDGS
import re
import markdownify
import httpx
import readabilipy
import json
import math

# --- Load environment variables ---
load_dotenv()

TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")
USER_AGENT = "Mozilla/5.0 (compatible; HealthBot/1.0; +https://example.com/bot)"

# MET values for various exercises (approximate)
EXERCISE_MET = {
    "walking (3 mph)": 3.5,
    "jogging (5 mph)": 7.0,
    "cycling (moderate)": 6.8,
    "running": 9.0,
    "jump rope": 10.0
}

assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"

# --- Auth Provider ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="puch-client",
                scopes=["*"],
                expires_at=None,
            )
        return None

# --- Rich Tool Description model ---
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None

# --- Fetch Utility Class ---
class Fetch:
    USER_AGENT = "Puch/1.0 (Autonomous)"

    @classmethod
    async def fetch_url(
        cls,
        url: str,
        user_agent: str,
        force_raw: bool = False,
    ) -> tuple[str, str]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    follow_redirects=True,
                    headers={"User-Agent": user_agent},
                    timeout=30,
                )
            except httpx.HTTPError as e:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"))

            if response.status_code >= 400:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url} - status code {response.status_code}"))

            page_raw = response.text

        content_type = response.headers.get("content-type", "")
        is_page_html = "text/html" in content_type

        if is_page_html and not force_raw:
            return cls.extract_content_from_html(page_raw), ""

        return (
            page_raw,
            f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n",
        )

    @staticmethod
    def extract_content_from_html(html: str) -> str:
        """Extract and convert HTML content to Markdown format."""
        ret = readabilipy.simple_json.simple_json_from_html_string(html, use_readability=True)
        if not ret or not ret.get("content"):
            return "<error>Page failed to be simplified from HTML</error>"
        content = markdownify.markdownify(ret["content"], heading_style=markdownify.ATX)
        return content

    # @staticmethod
    # async def google_search_links(query: str, num_results: int = 5) -> list[str]:
    #     """
    #     Perform a scoped DuckDuckGo search and return a list of job posting URLs.
    #     (Using DuckDuckGo because Google blocks most programmatic scraping.)
    #     """
    #     ddg_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
    #     links = []

    #     async with httpx.AsyncClient() as client:
    #         resp = await client.get(ddg_url, headers={"User-Agent": Fetch.USER_AGENT})
    #         if resp.status_code != 200:
    #             return ["<error>Failed to perform search.</error>"]

    #     from bs4 import BeautifulSoup
    #     soup = BeautifulSoup(resp.text, "html.parser")
    #     for a in soup.find_all("a", class_="result__a", href=True):
    #         href = a["href"]
    #         if "http" in href:
    #             links.append(href)
    #         if len(links) >= num_results:
    #             break

    #     return links or ["<error>No results found.</error>"]



# --- MCP Server Setup ---
mcp = FastMCP(
    "Job Finder MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)

# --- Tool: validate (required by Puch) ---
@mcp.tool
async def validate() -> str:
    return MY_NUMBER


def compute_bmi(weight_kg: float, height_cm: float) -> Dict[str, float | str]:
    if height_cm <= 0 or weight_kg <= 0:
        return {"error": "Height and weight must be positive numbers."}
    
    h_m = height_cm / 100.0
    bmi = round(weight_kg / (h_m ** 2), 2)
    
    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal weight"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"
    
    return {"bmi": bmi, "category": category}

@mcp.tool()
async def generate_bmi(weight_kg: float, height_cm: float) -> Dict[str, float | str]:
    """
    Calculate BMI from weight (kg) and height (cm), and return BMI value with category.
    """
    return compute_bmi(weight_kg, height_cm)



def compute_bmi_and_calories(weight_kg: float, height_cm: float) -> Dict[str, float | str]:
    if height_cm <= 0 or weight_kg <= 0:
        return {"error": "Height and weight must be positive numbers."}

    # Calculate BMI
    h_m = height_cm / 100.0
    bmi = round(weight_kg / (h_m ** 2), 2)

    # Determine category and calorie limit
    if bmi < 18.5:
        category = "Underweight"
        calories = 2500  # general gain suggestion
    elif bmi < 25:
        category = "Normal weight"
        calories = 2000  # maintenance suggestion
    elif bmi < 30:
        category = "Overweight"
        calories = 1800  # moderate deficit
    else:
        category = "Obese"
        calories = 1500  # aggressive but safe deficit

    return {
        "bmi": bmi,
        "category": category,
        "suggested_calories_per_day": calories
    }

@mcp.tool()
async def bmi_calorie_limit(weight_kg: float, height_cm: float) -> Dict[str, float | str]:
    """
    Calculate BMI from weight (kg) and height (cm), return BMI, category, and a suggested daily calorie limit.
    """
    return compute_bmi_and_calories(weight_kg, height_cm)



def estimate_exercise_durations(calories, weight_kg):
    """
    Estimate the duration of various exercises required to burn a given number of calories.

    Formula:
        Calories burned per minute = (MET × weight_kg × 3.5) / 200

    Args:
        calories (float or int): Total calories to burn.
        weight_kg (float or int): Weight of the person in kilograms.

    Returns:
        dict: A dictionary mapping exercise names (str) to the estimated duration (int, in minutes).
    """
    durations = {}
    for activity, met in EXERCISE_MET.items():
        cal_per_min = (float(met) * float(weight_kg) * 3.5) / 200
        minutes_needed = round(calories / cal_per_min)
        durations[activity] = minutes_needed
    return durations


def fetch_calories(meal_name):
    """
    Fetch the approximate calorie count for a given meal using DuckDuckGo Search.

    Args:
        meal_name (str): Name of the meal or food item to search for.

    Returns:
        int or None: Estimated calorie value if found, else None.
    """
    query = f"{meal_name} calories"
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=3)
        for result in results:
            text = result.get("body", "") + " " + result.get("title", "")
            match = re.search(r"(\d{2,5})\s*(kcal|calories?)", text, re.IGNORECASE)
            if match:
                return int(match.group(1))
    return None


@mcp.tool()
def cheat_meal_assist(meal, weight_kg):
    """
    Provide exercise recommendations to burn off the calories from a cheat meal.

    Args:
        meal (str): Name of the cheat meal.
        weight_kg (float or int): Weight of the person in kilograms.

    Returns:
        dict: Dictionary with:
              - 'calories' (int): Estimated calories of the cheat meal.
              - 'Activity_list_to_burn_calories' (dict): Mapping of activity to duration in minutes.
    """
    calories = fetch_calories(meal)
    if not calories:
        print("⚠️ Could not find calorie information.")
        return

    print(f"Cheat Meal: {meal}")
    print(f"🔥 Estimated Calories: {calories} kcal\n")

    suggestions = estimate_exercise_durations(calories, weight_kg)
    print("🏃 To burn it off, you could do:")
    for activity, minutes in suggestions.items():
        print(f" - {activity}: {minutes} minutes")
    return {"calories": calories, "Activity_list_to_burn_calories": suggestions}



# Image inputs and sending images

MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION = RichToolDescription(
    description="Convert an image to black and white and save it.",
    use_when="Use this tool when the user provides an image URL and requests it to be converted to black and white.",
    side_effects="The image will be processed and saved in a black and white format.",
)

@mcp.tool(description=MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION.model_dump_json())
async def make_img_black_and_white(
    puch_image_data: Annotated[str, Field(description="Base64-encoded image data to convert to black and white")] = None,
) -> list[TextContent | ImageContent]:
    import base64
    import io

    from PIL import Image

    try:
        image_bytes = base64.b64decode(puch_image_data)
        image = Image.open(io.BytesIO(image_bytes))

        bw_image = image.convert("L")

        buf = io.BytesIO()
        bw_image.save(buf, format="PNG")
        bw_bytes = buf.getvalue()
        bw_base64 = base64.b64encode(bw_bytes).decode("utf-8")

        return [ImageContent(type="image", mimeType="image/png", data=bw_base64)]
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))

# --- Run MCP Server ---
async def main():
    print("🚀 Starting MCP server on http://0.0.0.0:8086")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())
