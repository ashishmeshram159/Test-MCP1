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


# # ---------- helpers ----------
# def compute_bmi_category(weight_kg: float, height_cm: float) -> Dict[str, Any]:
#     if height_cm <= 0 or weight_kg <= 0:
#         return {"error": "Height/weight must be positive."}
#     h = height_cm / 100.0
#     bmi = round(weight_kg / (h * h), 2)
#     if bmi < 18.5:
#         cat = "Underweight"
#     elif bmi < 25:
#         cat = "Normal weight"
#     elif bmi < 30:
#         cat = "Overweight"
#     else:
#         cat = "Obesity"
#     return {"bmi": bmi, "category": cat}

# def mifflin_st_jeor(weight_kg: float, height_cm: float, age: int, sex: str) -> float:
#     if sex.lower().startswith("m"):
#         return 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
#     return 10 * weight_kg + 6.25 * height_cm - 5 * age - 161

# ACTIVITY = {
#     "sedentary": 1.2,
#     "light": 1.375,
#     "moderate": 1.55,
#     "active": 1.725,
#     "very_active": 1.9,
# }

# async def call_llm_chat(
#     messages: List[Dict[str, str]],
#     model: str,
#     api_base: str,
#     api_key: str,
#     temperature: float = 0.2,
#     timeout: int = 60,
# ) -> str:
#     async with httpx.AsyncClient(timeout=timeout) as client:
#         r = await client.post(
#             api_base,
#             headers={
#                 "Authorization": f"Bearer {api_key}",
#                 "Content-Type": "application/json",
#                 "User-Agent": "HealthMCP/1.0",
#             },
#             json={"model": model, "temperature": temperature, "messages": messages},
#         )
#     r.raise_for_status()
#     data = r.json()
#     return data["choices"][0]["message"]["content"]

# def parse_json_loose(s: str) -> Dict[str, Any]:
#     s = s.strip()
#     if s.startswith("```"):
#         s = s.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
#     return json.loads(s)

# # ---------- tools ----------
# @mcp.tool(
#     name="calculate_bmi_and_category",
#     description="Compute BMI and WHO category from weight (kg) and height (cm).",
#     schema={
#         "type": "object",
#         "properties": {
#             "weight_kg": {"type": "number", "minimum": 1},
#             "height_cm": {"type": "number", "minimum": 1},
#         },
#         "required": ["weight_kg", "height_cm"],
#         "additionalProperties": False,
#     },
# )
# async def calculate_bmi_and_category(weight_kg: float, height_cm: float) -> Dict[str, Any]:
#     return compute_bmi_category(weight_kg, height_cm)

# @mcp.tool(
#     name="llm_diet_recommendation",
#     description="Personalized 1-day diet plan as JSON (calories/macros/meal plan/grocery/cautions), using BMI+BMR+TDEE plus an LLM.",
#     schema={
#         "type": "object",
#         "properties": {
#             "weight_kg": {"type": "number", "minimum": 1},
#             "height_cm": {"type": "number", "minimum": 1},
#             "age": {"type": "integer", "minimum": 10, "maximum": 100},
#             "sex": {"type": "string", "enum": ["male", "female"]},
#             "activity_level": {
#                 "type": "string",
#                 "enum": ["sedentary", "light", "moderate", "active", "very_active"],
#             },
#             "goal": {"type": "string", "enum": ["lose", "maintain", "gain"]},
#             "dietary_prefs": {"type": "array", "items": {"type": "string"}},
#             "allergies": {"type": "array", "items": {"type": "string"}},
#             "model": {"type": "string", "default": "gpt-4o-mini"},
#             "provider_api_base": {
#                 "type": "string",
#                 "default": "https://api.openai.com/v1/chat/completions",
#             },
#             "api_key_env": {"type": "string", "default": "OPENAI_API_KEY"},
#             "temperature": {"type": "number", "default": 0.2},
#         },
#         "required": ["weight_kg", "height_cm", "age", "sex", "activity_level", "goal"],
#         "additionalProperties": False,
#     },
# )
# async def llm_diet_recommendation(
#     weight_kg: float,
#     height_cm: float,
#     age: int,
#     sex: str,
#     activity_level: str,
#     goal: str,
#     dietary_prefs: Optional[List[str]] = None,
#     allergies: Optional[List[str]] = None,
#     model: str = "gpt-4o-mini",
#     provider_api_base: str = "https://api.openai.com/v1/chat/completions",
#     api_key_env: str = "OPENAI_API_KEY",
#     temperature: float = 0.2,
# ) -> Dict[str, Any]:
#     basics = compute_bmi_category(weight_kg, height_cm)
#     if "error" in basics:
#         return basics

#     bmr = mifflin_st_jeor(weight_kg, height_cm, age, sex)
#     tdee = bmr * ACTIVITY.get(activity_level, 1.375)

#     if goal == "lose":
#         target_cal = max(1200, tdee - 400)
#         protein_g = round(1.6 * weight_kg)
#     elif goal == "gain":
#         target_cal = tdee + 300
#         protein_g = round(1.8 * weight_kg)
#     else:
#         target_cal = tdee
#         protein_g = round(1.4 * weight_kg)

#     target_cal = int(round(target_cal))
#     cal_p = int(protein_g * 4)
#     remaining = max(0, target_cal - cal_p)
#     carbs_cal = int(remaining * 0.6)
#     fat_cal = remaining - carbs_cal
#     carbs_g = round(carbs_cal / 4)
#     fat_g = round(fat_cal / 9)

#     api_key = os.getenv(api_key_env)
#     if not api_key:
#         return {"error": f"{api_key_env} not set in environment."}

#     system_msg = (
#         "You are a precise nutrition coach. Respond ONLY with valid minified JSON matching the schema."
#     )
#     prefs = ", ".join(dietary_prefs or []) or "none"
#     allergy_str = ", ".join(allergies or []) or "none"

#     schema_block = """
# {"calories_target": int, "macros": {"protein_g": int, "carbs_g": int, "fat_g": int},
# "meal_plan": {"breakfast":[{"item":str,"portion":str,"calories":int}],
# "lunch":[{"item":str,"portion":str,"calories":int}],
# "snack":[{"item":str,"portion":str,"calories":int}],
# "dinner":[{"item":str,"portion":str,"calories":int}]},
# "grocery_list":[str], "cautions":[str], "motivation_tips":[str]}
# """.strip()

#     user_msg = f"""
# Return JSON for a 1-day diet plan tailored to:

# Sex:{sex}; Age:{age}; Height_cm:{height_cm}; Weight_kg:{weight_kg};
# BMI:{basics.get("bmi")}; BMI_Category:{basics.get("category")};
# Activity_Level:{activity_level}; Goal:{goal};
# Dietary_Preferences:{prefs}; Allergies:{allergy_str};
# Target_Calories:{target_cal}

# Schema:
# {schema_block}

# Constraints:
# - Respect preferences/allergies.
# - Meal calories sum within ±10% of calories_target.
# - Prefer high-fiber carbs, lean proteins; moderate sugar/sodium.
# - Use commonly available foods in India when possible.
# """.strip()

#     assistant_hint = json.dumps(
#         {"suggested_macros_hint": {"protein_g": int(protein_g), "carbs_g": int(carbs_g), "fat_g": int(fat_g)}},
#         separators=(",", ":"),
#     )

#     content = await call_llm_chat(
#         messages=[
#             {"role": "system", "content": system_msg},
#             {"role": "assistant", "content": assistant_hint},
#             {"role": "user", "content": user_msg},
#         ],
#         model=model,
#         api_base=provider_api_base,
#         api_key=api_key,
#         temperature=temperature,
#     )

#     try:
#         plan = parse_json_loose(content)
#     except Exception as e:
#         return {"error": "Model did not return valid JSON.", "details": str(e), "raw": content}

#     plan["_meta"] = {
#         "bmi": basics["bmi"],
#         "bmi_category": basics["category"],
#         "bmr": int(round(bmr)),
#         "tdee": int(round(tdee)),
#         "target_calories": target_cal,
#         "suggested_macros_hint": {"protein_g": int(protein_g), "carbs_g": int(carbs_g), "fat_g": int(fat_g)},
#     }
#     return plan

# @mcp.tool(
#     name="llm_exercise_suggestions",
#     description="Get a JSON weekly exercise split for a goal (beginner–intermediate).",
#     schema={
#         "type": "object",
#         "properties": {
#             "goal": {"type": "string"},
#             "constraints": {"type": "array", "items": {"type": "string"}},
#             "model": {"type": "string", "default": "gpt-4o-mini"},
#             "provider_api_base": {
#                 "type": "string",
#                 "default": "https://api.openai.com/v1/chat/completions",
#             },
#             "api_key_env": {"type": "string", "default": "OPENAI_API_KEY"},
#         },
#         "required": ["goal"],
#         "additionalProperties": False,
#     },
# )
# async def llm_exercise_suggestions(
#     goal: str,
#     constraints: Optional[List[str]] = None,
#     model: str = "gpt-4o-mini",
#     provider_api_base: str = "https://api.openai.com/v1/chat/completions",
#     api_key_env: str = "OPENAI_API_KEY",
# ) -> Dict[str, Any]:
#     api_key = os.getenv(api_key_env)
#     if not api_key:
#         return {"error": f"{api_key_env} not set in environment."}

#     system_msg = "You are a concise fitness coach. Return ONLY valid JSON."
#     user_msg = f"""
# Goal: {goal}
# Constraints: {", ".join(constraints or []) or "none"}

# Schema:
# {{"weekly_plan": {{"Mon":[{{"exercise":str,"sets":int,"reps_or_time":str}}],
# "Tue":[{{"exercise":str,"sets":int,"reps_or_time":str}}],"Wed":[{{"exercise":str,"sets":int,"reps_or_time":str}}],
# "Thu":[{{"exercise":str,"sets":int,"reps_or_time":str}}],"Fri":[{{"exercise":str,"sets":int,"reps_or_time":str}}],
# "Sat":[{{"exercise":str,"sets":int,"reps_or_time":str}}],"Sun":[{{"exercise":str,"sets":int,"reps_or_time":str}}]}},
# "notes":[str]}}
# Keep it realistic for beginner–intermediate with limited equipment if unspecified.
# """.strip()

#     content = await call_llm_chat(
#         messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
#         model=model,
#         api_base=provider_api_base,
#         api_key=api_key,
#         temperature=0.3,
#     )

#     try:
#         return parse_json_loose(content)
#     except Exception as e:
#         return {"error": "Model did not return valid JSON.", "details": str(e), "raw": content}




# # --- Tool: job_finder (now smart!) ---
# JobFinderDescription = RichToolDescription(
#     description="Smart job tool: analyze descriptions, fetch URLs, or search jobs based on free text.",
#     use_when="Use this to evaluate job descriptions or search for jobs using freeform goals.",
#     side_effects="Returns insights, fetched job descriptions, or relevant job links.",
# )

# @mcp.tool(description=JobFinderDescription.model_dump_json())
# async def job_finder(
#     user_goal: Annotated[str, Field(description="The user's goal (can be a description, intent, or freeform query)")],
#     job_description: Annotated[str | None, Field(description="Full job description text, if available.")] = None,
#     job_url: Annotated[AnyUrl | None, Field(description="A URL to fetch a job description from.")] = None,
#     raw: Annotated[bool, Field(description="Return raw HTML content if True")] = False,
# ) -> str:
#     """
#     Handles multiple job discovery methods: direct description, URL fetch, or freeform search query.
#     """
#     if job_description:
#         return (
#             f"📝 **Job Description Analysis**\n\n"
#             f"---\n{job_description.strip()}\n---\n\n"
#             f"User Goal: **{user_goal}**\n\n"
#             f"💡 Suggestions:\n- Tailor your resume.\n- Evaluate skill match.\n- Consider applying if relevant."
#         )

#     if job_url:
#         content, _ = await Fetch.fetch_url(str(job_url), Fetch.USER_AGENT, force_raw=raw)
#         return (
#             f"🔗 **Fetched Job Posting from URL**: {job_url}\n\n"
#             f"---\n{content.strip()}\n---\n\n"
#             f"User Goal: **{user_goal}**"
#         )

#     if "look for" in user_goal.lower() or "find" in user_goal.lower():
#         links = await Fetch.google_search_links(user_goal)
#         return (
#             f"🔍 **Search Results for**: _{user_goal}_\n\n" +
#             "\n".join(f"- {link}" for link in links)
#         )

#     raise McpError(ErrorData(code=INVALID_PARAMS, message="Please provide either a job description, a job URL, or a search query in user_goal."))


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
