# -*- coding: utf-8 -*-
"""
巩固提取器（优化版）

架构：
1. 借鉴 GraphRAG 的两阶段提取 + Few-shot 示例
2. 使用 Instructor 保证输出格式稳定
3. 统一规则，简化流程

流程：
阶段1：提取实体 → List[Entity]
阶段2：基于实体提取关系 → List[Relationship]
"""

import logging
import os
import re
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)
_DEBUG = os.getenv("DEBUG_CONSOLIDATED_EXTRACTOR", "0") == "1"


# =============================================================================
# Pydantic Schema
# =============================================================================

class Entity(BaseModel):
    """实体"""
    name: str = Field(description="实体名称，首字母大写")
    type: str = Field(description="实体类型，如 Person, Organization, Location, Event, Application, Item, Concept 等")
    description: str = Field(default="", description="实体描述")
    
    @field_validator("type")
    @classmethod
    def validate_type(cls, v):
        # 泛化设计：不限制类型，只做标准化
        if not v or not v.strip():
            return "Concept"
        
        # 标准化：首字母大写
        v = v.strip().title().replace(" ", "")
        
        # 常见类型映射（可选，帮助 LLM 输出更一致）
        type_mapping = {
            "User": "Person",
            "Character": "Person",
            "Company": "Organization",
            "Museum": "Organization",
            "Restaurant": "Organization",
            "Hotel": "Organization",
            "Venue": "Organization",
            "City": "Location",
            "Country": "Location",
            "Place": "Location",
            "Geo": "Location",
            "Exhibition": "Event",
            "Tour": "Event",
            "Lecture": "Event",
            "Meeting": "Event",
            "App": "Application",
            "Software": "Application",
            "Tool": "Application",
            "Service": "Application",
            "Object": "Item",
            "Product": "Item",
            "Card": "Item",
            "Device": "Item",
            "Number": "Value",
            "Count": "Value",
            "Amount": "Value",
            "Idea": "Concept",
            "Hobby": "Concept",
            "Skill": "Concept",
        }
        
        return type_mapping.get(v, v)  # 如果没有映射，保留原始类型


class Relationship(BaseModel):
    """关系"""
    source: str = Field(description="源实体名称，必须是已提取的实体")
    target: str = Field(description="目标实体名称，必须是已提取的实体")
    type: str = Field(description="关系类型，大写，如 VISITED, LIVES_IN, WORKS_AT")
    description: str = Field(default="", description="关系描述")
    strength: float = Field(default=8.0, description="关系强度 1-10")
    
    @field_validator("type")
    @classmethod
    def validate_type(cls, v):
        # 确保大写
        return v.upper().replace(" ", "_")


class EntityExtractionResult(BaseModel):
    """阶段1：实体提取结果"""
    entities: List[Entity] = Field(default_factory=list)


class RelationshipExtractionResult(BaseModel):
    """阶段2：关系提取结果"""
    relationships: List[Relationship] = Field(default_factory=list)


# =============================================================================
# Instructor 客户端
# =============================================================================

_INSTRUCTOR_AVAILABLE = False
_instructor_client = None
_instructor_model = None

try:
    import instructor
    from openai import OpenAI
    _INSTRUCTOR_AVAILABLE = True
except ImportError:
    logger.warning("[consolidated_extractor] instructor not installed, run: pip install instructor")


def _get_instructor_client():
    """获取 Instructor 客户端（单例）"""
    global _instructor_client, _instructor_model
    
    if not _INSTRUCTOR_AVAILABLE:
        return None, None
    
    if _instructor_client is not None:
        return _instructor_client, _instructor_model
    
    try:
        from config import (
            CHEAP_GRAPHRAG_API_BASE,
            CHEAP_GRAPHRAG_CHAT_API_KEY,
            CHEAP_GRAPHRAG_CHAT_MODEL,
        )
        import httpx
        
        # 设置超时：连接 30s，读取 120s，总计 180s
        timeout = httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0)
        http_client = httpx.Client(timeout=timeout)
        
        client = OpenAI(
            base_url=CHEAP_GRAPHRAG_API_BASE,
            api_key=CHEAP_GRAPHRAG_CHAT_API_KEY,
            http_client=http_client,
            max_retries=3,  # OpenAI 客户端内置重试
        )
        _instructor_client = instructor.from_openai(client)
        _instructor_model = CHEAP_GRAPHRAG_CHAT_MODEL
        
        if _DEBUG:
            logger.info(f"[consolidated_extractor] initialized with model: {_instructor_model}, timeout=180s, max_retries=3")
        
        return _instructor_client, _instructor_model
    except Exception as e:
        logger.error(f"[consolidated_extractor] failed to init client: {e}")
        return None, None


# =============================================================================
# 阶段0：提取简单事实（精炼信息，不做推理）
# =============================================================================

SIMPLE_FACTS_EXTRACTION_PROMPT = """You are extracting simple facts from a conversation between USER and ASSISTANT.

-Goal-
Extract factual information from BOTH USER and ASSISTANT. Each fact must be self-contained and preserve ALL important details.

-Core Rules-
1. Extract from BOTH USER and ASSISTANT, mark source as "user" or "assistant"
2. Each fact must be COMPLETE - preserve ALL details (time, location, names, numbers, prices)
3. Do NOT extract: hypotheticals, questions, vague statements
4. NUMBERED LISTS: Always preserve position with "#N" prefix (e.g., "#7 Transcriptionist")
5. MULTIPLE ITEMS: Extract EACH item as a SEPARATE fact (for counting questions)
6. SPECIFIC NAMES: Always preserve exact product/brand/venue names (e.g., "Mod Podge", "Seattle International Film Festival")

-Examples-

Example 1 (Numbered list - preserve position):
USER: What are some healthy breakfast options?
ASSISTANT: Here are some ideas:
1. Greek yogurt with berries
2. Oatmeal with nuts
3. Avocado toast
4. Smoothie bowl
5. Scrambled eggs

Facts:
- [user] User asked about healthy breakfast options
- [assistant] #1 Greek yogurt with berries is suggested as a healthy breakfast option
- [assistant] #2 Oatmeal with nuts is suggested as a healthy breakfast option
- [assistant] #3 Avocado toast is suggested as a healthy breakfast option
- [assistant] #4 Smoothie bowl is suggested as a healthy breakfast option
- [assistant] #5 Scrambled eggs is suggested as a healthy breakfast option
(The "#N" prefix is ESSENTIAL for "what was the Nth item" questions!)

Example 2 (Specific product/brand names):
USER: How do I remove stains from my white shirt?
ASSISTANT: Try using OxiClean or Shout stain remover. Apply it directly to the stain, let it sit for 10 minutes, then wash normally.

Facts:
- [user] User wants to remove stains from their white shirt
- [assistant] OxiClean is recommended for removing stains from white shirts
- [assistant] Shout stain remover is recommended for removing stains
- [assistant] Apply stain remover directly to the stain and let it sit for 10 minutes before washing
(Specific product names like "OxiClean" and "Shout" MUST be preserved!)

Example 3 (Event venue/location):
USER: I met the author at a book signing event at Barnes & Noble last Saturday. She signed my copy of her new novel.

Facts:
- [user] User attended a book signing event at Barnes & Noble last Saturday
- [user] User met the author at Barnes & Noble
- [user] User got their copy of the author's new novel signed
("at Barnes & Noble" MUST be preserved - extract venue as part of the fact!)

Example 4 (Multiple items - extract separately):
USER: I picked up some groceries yesterday. I got apples, bananas, and oranges from the farmer's market.

Facts:
- [user] User bought apples from the farmer's market yesterday
- [user] User bought bananas from the farmer's market yesterday
- [user] User bought oranges from the farmer's market yesterday
(Each item MUST be a separate fact for counting questions!)

Example 5 (Price and numbers):
USER: I found a great deal on a jacket. It was originally $80 but I got it for $60.

Facts:
- [user] User found a deal on a jacket
- [user] User's jacket was originally priced at $80
- [user] User bought the jacket for $60
(BOTH prices MUST be preserved for discount calculation!)

-Text to extract from-
{text}

-Reference time (CRITICAL for event_time calculation)-
{session_time}

-EVENT TIME CALCULATION (CRITICAL - MUST DO FOR EVERY FACT WITH TIME EXPRESSION)-
For EVERY fact that contains a time expression, you MUST calculate the ACTUAL DATE.

**CRITICAL: If a fact mentions ANY time expression (yesterday, last week, a few years ago, etc.), you MUST calculate event_time!**

**How to calculate event_time:**
1. Parse the reference time above (e.g., "2024-05-24T10:00:00" means today is 2024-05-24)
2. For relative time expressions, calculate the actual date:
   - "today" / "just" / "just now" → same as reference date (2024-05-24)
   - "yesterday" → reference date - 1 day (2024-05-23)
   - "3 days ago" → reference date - 3 days (2024-05-21)
   - "last week" → reference date - 7 days (2024-05-17)
   - "last month" → reference date - 30 days (2024-04-24)
   - "last year" / "last summer" → reference date - 365 days (2023-05-24)
   - "a few years ago" → reference date - 3 years (2021-05-24)
   - "several years ago" → reference date - 4 years (2020-05-24)
   - "many years ago" → reference date - 5+ years (2019-05-24)
3. For specific dates mentioned, use that date directly
4. For ongoing states or habits (no specific time), leave event_time empty
5. For future events, calculate forward from reference date

**Examples with reference time 2024-05-24:**
- "User bought shoes today" → event_time: "2024-05-24"
- "User visited Tokyo yesterday" → event_time: "2024-05-23"
- "User started a new job last month" → event_time: "2024-04-24"
- "User graduated last year" → event_time: "2023-05-24"
- "User traveled to Paris a few years ago" → event_time: "2021-05-24" (CRITICAL: "a few years ago" = 3 years!)
- "User went on a road trip a few years ago" → event_time: "2021-05-24" (CRITICAL: MUST calculate!)
- "User visited the Grand Canyon a few years ago" → event_time: "2021-05-24" (CRITICAL: MUST calculate!)
- "User practices guitar daily" → event_time: "" (ongoing habit, no specific date)
- "User is planning to visit Kyoto next week" → event_time: "2024-05-31"

**CRITICAL REMINDER:**
- "a few years ago" ALWAYS means ~3 years before reference time
- "last summer" ALWAYS means ~1 year before reference time
- If the fact text contains "a few years ago", "last year", "last summer", etc., you MUST provide event_time!

Extract 5-20 simple facts. Each fact must be self-contained with all relevant context (time, location, people, purpose). Mark each with [user] or [assistant] prefix."""


class SimpleFact(BaseModel):
    """单条简单事实"""
    text: str = Field(description="事实内容，完整保留所有细节")
    source: str = Field(description="来源：user 或 assistant")
    event_time: str = Field(default="", description="事件发生的实际日期，格式 YYYY-MM-DD。根据 reference time 和时间表达式计算。如果是持续状态或无法确定则留空。")


class SimpleFactsResult(BaseModel):
    """简单事实提取结果"""
    facts: List[SimpleFact] = Field(default_factory=list, description="简单事实列表")


def extract_simple_facts(
    text: str,
    session_time_iso: str = "",
) -> List[Dict[str, str]]:
    """
    阶段0：从对话中提取简单事实（精炼信息）
    
    Args:
        text: 对话文本（USER + ASSISTANT）
        session_time_iso: 会话时间
    
    Returns:
        简单事实列表，每条包含 {"text": "...", "source": "user/assistant", "event_time": "YYYY-MM-DD"}
    """
    if not text or not text.strip():
        return []
    
    if not _INSTRUCTOR_AVAILABLE:
        logger.warning("[consolidated_extractor] instructor not available")
        return []
    
    client, model = _get_instructor_client()
    if not client:
        return []
    
    prompt = SIMPLE_FACTS_EXTRACTION_PROMPT.format(
        text=text,
        session_time=session_time_iso or "unknown",
    )
    
    try:
        result = client.chat.completions.create(
            model=model,
            response_model=SimpleFactsResult,
            messages=[{"role": "user", "content": prompt}],
            max_retries=2,
        )
        
        facts = []
        for f in (result.facts or []):
            fact_text = f.text.strip() if hasattr(f, 'text') else str(f).strip()
            fact_source = f.source.lower() if hasattr(f, 'source') else "user"
            fact_event_time = f.event_time.strip() if hasattr(f, 'event_time') else ""
            
            # 清理 [user] 或 [assistant] 前缀（如果 LLM 在 text 中也加了）
            if fact_text.startswith("[user]"):
                fact_text = fact_text[6:].strip()
                fact_source = "user"
            elif fact_text.startswith("[assistant]"):
                fact_text = fact_text[11:].strip()
                fact_source = "assistant"
            
            # 验证 event_time 格式（YYYY-MM-DD）
            if fact_event_time:
                import re
                if not re.match(r"^\d{4}-\d{2}-\d{2}$", fact_event_time):
                    # 尝试提取日期部分
                    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", fact_event_time)
                    fact_event_time = date_match.group(1) if date_match else ""
            
            if fact_text:
                facts.append({
                    "text": fact_text,
                    "source": fact_source,
                    "event_time": fact_event_time,
                })
        
        if _DEBUG:
            logger.info(f"[consolidated_extractor] extracted {len(facts)} simple facts")
            for f in facts:
                event_time_str = f" [event_time={f['event_time']}]" if f.get('event_time') else ""
                logger.info(f"  - [{f['source']}]{event_time_str} {f['text']}")
        
        return facts
    
    except Exception as e:
        logger.error(f"[consolidated_extractor] simple facts extraction failed: {e}")
        return []




# =============================================================================
# 阶段1：实体提取 Prompt（借鉴 GraphRAG + Few-shot）
# =============================================================================

ENTITY_EXTRACTION_PROMPT = """You are extracting entities from a conversation.

-Goal-
Identify ALL important entities (people, organizations, locations, events) from the text.

-Steps-
1. Read the text carefully
2. Identify all entities with their types:
   - Person: people, users (use "User" for the speaker)
   - Organization: companies, institutions, venues, establishments (places you can visit)
   - Location: cities, countries, geographic areas
   - Event: exhibitions, tours, lectures, meetings (things that happen at a specific time)
   - Application: apps, software, tools (TripIt, Google Maps, Uber)
   - Item: physical objects, products, cards (Suica card, camera, bike)
   - Concept: abstract ideas, skills, hobbies (Buddhism, Tennis, Photography)
   - (Use other types if needed, e.g., Vehicle, Food, Book, etc.)
3. For each entity, provide a brief description

-CRITICAL Rules-
1. Extract ALL named entities mentioned in the text - do not miss any!
2. If user says "I visited X" or "I went to X", X is likely an Organization
3. Physical places you can visit (museums, restaurants, hotels, galleries, theaters) are Organizations
4. Time-bound activities (exhibitions, tours, lectures, conferences) are Events
5. Use "User" for the person speaking in first person ("I", "my")
6. When text mentions "X Museum's Y exhibition", extract BOTH:
   - X Museum as Organization
   - Y exhibition as Event
7. Apps and software are Application type (TripIt, Google Maps, Uber, etc.)
8. Physical items like cards, devices, products are Item type (Suica card, camera, bike, etc.)
9. You can use custom entity types if the standard ones don't fit well

-Entity Type Guidelines-
Organization (places you can visit):
- Museums: Science Museum, Natural History Museum, Museum of Contemporary Art, etc.
- Galleries, theaters, cinemas
- Restaurants, cafes, hotels, shops
- Companies, schools, hospitals
- Any named venue or establishment

Event (time-bound activities):
- Exhibitions: "Space Exploration exhibition", "Dinosaur Fossils exhibition"
- Shows, performances, tours
- Lectures, conferences, seminars
- Meetings, parties, ceremonies

Location (geographic):
- Cities, countries, regions
- Parks, mountains, rivers (natural places)

Application (apps and software):
- Mobile apps: TripIt, Google Maps, Uber, Airbnb
- Software: Excel, Photoshop
- Online services: Netflix, Spotify

Item (physical objects):
- Cards: Suica card, credit card, membership card
- Devices: camera, phone, laptop
- Products: bike, car, book

Concept (abstract):
- Skills: Python, cooking, photography
- Hobbies: tennis, hiking, reading
- Ideas: Buddhism, minimalism

-Examples-

Example 1:
Text: "I visited the Science Museum today with my colleague David."
Output:
- User (Person): The speaker
- Science Museum (Organization): A museum the user visited
- David (Person): The user's colleague

Example 2:
Text: "I visited the Science Museum's Space Exploration exhibition with David."
Output:
- User (Person): The speaker
- Science Museum (Organization): The museum where the exhibition is held
- Space Exploration exhibition (Event): An exhibition at the Science Museum
- David (Person): Person who went with the user

Example 3:
Text: "I attended a lecture series at the Museum of Contemporary Art where Dr. Rodriguez spoke."
Output:
- User (Person): The speaker
- Museum of Contemporary Art (Organization): The venue where the lecture was held
- Lecture series (Event): The event the user attended
- Dr. Rodriguez (Person): The speaker at the lecture

Example 5 (CRITICAL - "attended X at Y" pattern):
Text: "I attended a lectures series at the Museum of Contemporary Art recently, where Dr. Maria Rodriguez spoke about feminist art."
Output:
- User (Person): The speaker
- Museum of Contemporary Art (Organization): The venue - MUST be extracted as Organization
- Lectures series on feminist art (Event): The event the user attended
- Dr. Maria Rodriguez (Person): The speaker at the lecture

Example 4:
Text: "I went to the Modern Art Gallery and had dinner at Luigi's Restaurant."
Output:
- User (Person): The speaker
- Modern Art Gallery (Organization): A gallery the user visited
- Luigi's Restaurant (Organization): A restaurant where the user had dinner

-Text to extract from-
{text}

-Reference time-
{session_time}
"""


# =============================================================================
# 阶段2：关系提取 Prompt（借鉴 GraphRAG + Few-shot）
# =============================================================================

RELATIONSHIP_EXTRACTION_PROMPT = """You are extracting relationships between entities.

-Goal-
Identify all relationships between the entities listed below.

-Entities (extracted in previous step)-
{entities_text}

-Rules-
1. source and target MUST be from the entity list above
2. Relationship type must be UPPERCASE with underscores (e.g., VISITED, WORKS_AT, HAS, USES)
3. Only extract relationships that are clearly stated or strongly implied in the text
4. Extract the relationship type that best describes the action/connection in the text
5. Be specific: prefer "DOWNLOADED" over "HAS" for apps, "STAYING_AT" over "VISITED" for hotels

-Common Relationship Patterns-
(These are examples, not a complete list. Use the verb/action from the text!)

POSSESSION: HAS, OWNS, BOUGHT, PURCHASED, GOT, RECEIVED, BORROWED, RENTED
USAGE: USES, DOWNLOADED, INSTALLED, RUNS, OPERATES
VISITS: VISITED, WENT_TO, ATTENDED, STAYED_AT, CHECKED_INTO
WORK: WORKS_AT, EMPLOYED_BY, MANAGES, LEADS
RESIDENCE: LIVES_IN, RESIDES_IN, MOVED_TO, RELOCATED_TO
TRAVEL: TRAVELING_TO, HEADING_TO, FLYING_TO, DRIVING_TO
PLANS: PLANS_TO_VISIT, WANTS_TO_BUY, INTENDS_TO, CONSIDERING
SOCIAL: KNOWS, MET, FRIENDS_WITH, MARRIED_TO, RELATED_TO
LOCATION: LOCATED_IN, LOCATED_AT, BASED_IN, SITUATED_IN
PREFERENCE: PREFERS, LIKES, LOVES, ENJOYS, FAVORS
CREATION: CREATED, MADE, BUILT, DESIGNED, WROTE

-Key Principle-
Extract the relationship type directly from the verb in the text:
- "I downloaded TripIt" → DOWNLOADED
- "I have a Suica card" → HAS
- "I'm staying at Park Hyatt" → STAYING_AT
- "I rented a car" → RENTED
- "I borrowed a book" → BORROWED

-Examples-

Example 1:
Entities: User (Person), TripIt (Application), Suica Card (Item), Tokyo (Location)
Text: "I'm heading to Tokyo soon. I just got a Suica card for transit. I also downloaded the TripIt app."
Output:
- User -[HEADING_TO]-> Tokyo: User is traveling to Tokyo (strength: 9)
- User -[HAS]-> Suica Card: User got a Suica card (strength: 9)
- User -[DOWNLOADED]-> TripIt: User downloaded TripIt app (strength: 9)

Example 2:
Entities: User (Person), Park Hyatt Tokyo (Organization), Rental Car (Item)
Text: "I'm staying at the Park Hyatt Tokyo. I also rented a car for the trip."
Output:
- User -[STAYING_AT]-> Park Hyatt Tokyo: User is staying at this hotel (strength: 9)
- User -[RENTED]-> Rental Car: User rented a car (strength: 9)

Example 3:
Entities: User (Person), Python (Concept), Coursera (Organization)
Text: "I'm learning Python through Coursera. I want to become a data scientist."
Output:
- User -[LEARNING]-> Python: User is learning Python (strength: 9)
- User -[USES]-> Coursera: User uses Coursera for learning (strength: 8)
- User -[WANTS_TO_BECOME]-> Data Scientist: User's career goal (strength: 8)

-Text-
{text}

-Reference time-
{session_time}
"""


# =============================================================================
# 阶段3：LLM 验证+补漏 Prompt
# =============================================================================

class ValidationResult(BaseModel):
    """验证结果"""
    missing_entities: List[Entity] = Field(default_factory=list, description="遗漏的实体")
    missing_relationships: List[Relationship] = Field(default_factory=list, description="遗漏的关系")
    entities_to_fix: List[Entity] = Field(default_factory=list, description="需要修正类型的实体")


VALIDATION_PROMPT = """You are a knowledge graph validator. Check if the extraction is complete and correct.

-Original Text-
{text}

-Extracted Entities-
{entities_text}

-Extracted Relationships-
{relationships_text}

-Your Task-
1. Check if any important entities are MISSING
2. Check if any important relationships are MISSING
3. Check if any entity types are WRONG
4. Check for CRITICAL patterns that are often missed (see below)

-Rules for Entity Types-
- Person: Individual people (User, Dr. Smith, my friend Alice)
- Organization: Physical places, companies, brands (museums, restaurants, Nike, Apple)
- Location: Geographic areas (cities, countries, parks, neighborhoods)
- Event: Time-bound activities (exhibitions, tours, lectures, appointments)
- Application: Apps, software, tools (TripIt, Google Maps, Uber, Excel)
- Item: Physical objects, cards, devices (Suica card, camera, bike, credit card)
- Value: Numbers, counts, amounts, ratios, durations, frequencies
- Concept: Abstract ideas, beliefs, hobbies, skills (Buddhism, Atheism, Tennis, Photography)

=== CRITICAL PATTERNS TO CHECK (VERY IMPORTANT) ===

0. APPS AND ITEMS (HIGHEST PRIORITY - often missed!):
   Look for: "downloaded", "have", "got", "use", "using", "card", "app"
   - "I downloaded TripIt" → MUST have: User -[DOWNLOADED]-> TripIt (Application)
   - "I have a Suica card" → MUST have: User -[HAS]-> Suica Card (Item)
   - "I use Google Maps" → MUST have: User -[USES]-> Google Maps (Application)
   - "I got a new camera" → MUST have: User -[HAS]-> Camera (Item)
   - If you see app/item usage but no relationship, ADD IT!

1. STATE CHANGES / PAST STATES:
   Look for: "used to be", "was previously", "was a", "switched from", "changed from", "no longer"
   - "I used to be a staunch atheist" → MUST have: User -[WAS]-> Atheist (Concept)
   - "I was previously a vegetarian" → MUST have: User -[WAS]-> Vegetarian (Concept)
   - "I switched from iPhone to Android" → MUST have: User -[WAS]-> iPhone User, User -[IS]-> Android User
   - If you see "used to be X" in text but no WAS relationship, ADD IT!

2. RATIOS AND PROPORTIONS:
   Look for: "ratio of", "X:Y", "X to Y", "X parts to Y parts", "dilution"
   - "dilution ratio of 1:10" → MUST have: Subject -[DILUTION_RATIO]-> 1:10 (Value)
   - "mix 1 part oil with 10 parts water" → MUST have: Oil -[MIX_RATIO]-> 1:10 (Value)
   - If you see a ratio in text but no ratio relationship, ADD IT!

3. FREQUENCIES:
   Look for: "every", "weekly", "daily", "twice a", "X times per", "once a"
   - "I play tennis every Sunday" → MUST have: User -[PLAYS]-> Tennis with frequency property OR User -[FREQUENCY]-> "every Sunday"
   - "We used to meet weekly, now we meet monthly" → MUST capture BOTH frequencies
   - If you see frequency info in text but no frequency relationship/property, ADD IT!

4. QUANTITIES AND COUNTS:
   Look for: numbers + nouns ("three bikes", "17 cameras", "5 appointments")
   - "I own three bikes" → MUST have: User -[OWNS_COUNT]-> 3 (Value) with entity="bikes"
   - "I have 17 cameras" → MUST have: User -[HAS_COUNT]-> 17 (Value) with entity="cameras"
   - If you see a count in text but no count relationship, ADD IT!

5. DURATIONS:
   Look for: "for X years/months/weeks", "since", "been doing X for"
   - "I've been collecting for three months" → MUST have: User -[DURATION]-> "three months" (Value)
   - "I've lived here since 2020" → MUST have: User -[LIVES_IN_SINCE]-> 2020 (Value)
   - If you see duration info but no duration relationship, ADD IT!

6. ENTITY ALIASES (same person/thing with different names):
   Look for: "my X (name)", "X, my Y", "also known as", "called"
   - "Dr. Johnson, my primary care physician" → MUST have: Dr. Johnson -[ALSO_KNOWN_AS]-> Primary Care Physician
   - "my friend Sarah (also called Sally)" → MUST have: Sarah -[ALSO_KNOWN_AS]-> Sally
   - If same entity has multiple names, ADD alias relationship!

7. PRICES AND COSTS:
   Look for: "$", "cost", "price", "paid", "spent"
   - "I paid $500 for the camera" → MUST have: Camera -[COST]-> $500 (Value)
   - "The renovation will cost $50,000" → MUST have: Renovation -[COST]-> $50,000 (Value)

8. APPOINTMENTS AND VISITS:
   Look for: "appointment with", "visited", "saw", "met with"
   - "I had an appointment with Dr. Smith" → MUST have: User -[HAD_APPOINTMENT_WITH]-> Dr. Smith
   - "I saw my dentist yesterday" → MUST have: User -[VISITED]-> Dentist

9. PREFERENCES AND OPINIONS:
   Look for: "prefer", "like", "love", "favorite", "enjoy"
   - "I prefer running on roads" → MUST have: User -[PREFERS]-> Running On Roads
   - "My favorite restaurant is X" → MUST have: User -[FAVORITE]-> X

10. PLANS AND INTENTIONS:
    Look for: "planning to", "going to", "will", "want to"
    - "I'm planning to visit Hawaii" → MUST have: User -[PLANS_TO_VISIT]-> Hawaii
    - "I want to learn guitar" → MUST have: User -[WANTS_TO_LEARN]-> Guitar

=== VENUE VISITS (existing rules) ===

When text says "I visited the X Museum's Y exhibition":
- MUST have: User -[VISITED]-> X Museum (Organization)
- SHOULD have: User -[ATTENDED]-> Y exhibition (Event)

When text says "I attended X at Y":
- MUST have: User -[VISITED]-> Y (Organization) - the venue
- SHOULD have: User -[ATTENDED]-> X (Event) - the activity

=== OUTPUT FORMAT ===

Return:
1. missing_entities: List of entities that should be added
2. missing_relationships: List of relationships that should be added
3. entities_to_fix: List of entities with corrected types

IMPORTANT: For each missing item, explain WHY it's missing (what pattern in the text was not captured).

If extraction is complete and correct, return empty lists.

-Reference time-
{session_time}
"""


# 是否启用 LLM 验证+补漏（可通过环境变量控制）
_ENABLE_LLM_VALIDATION = os.getenv("ENABLE_V2_LLM_VALIDATION", "1").strip().lower() in ("1", "true", "yes")


# =============================================================================
# 阶段1：提取实体
# =============================================================================

def extract_entities(
    text: str,
    session_time_iso: str = "",
) -> List[Entity]:
    """
    阶段1：从文本中提取实体
    
    Args:
        text: 对话文本
        session_time_iso: 会话时间
    
    Returns:
        实体列表
    """
    if not text or not text.strip():
        return []
    
    if not _INSTRUCTOR_AVAILABLE:
        logger.warning("[consolidated_extractor] instructor not available")
        return []
    
    client, model = _get_instructor_client()
    if not client:
        return []
    
    prompt = ENTITY_EXTRACTION_PROMPT.format(
        text=text,
        session_time=session_time_iso or "unknown",
    )
    
    try:
        result = client.chat.completions.create(
            model=model,
            response_model=EntityExtractionResult,
            messages=[{"role": "user", "content": prompt}],
            max_retries=2,
        )
        
        entities = result.entities or []
        
        if _DEBUG:
            logger.info(f"[consolidated_extractor] extracted {len(entities)} entities")
            for e in entities:
                logger.info(f"  - {e.name} ({e.type})")
        
        return entities
    
    except Exception as e:
        logger.error(f"[consolidated_extractor] entity extraction failed: {e}")
        return []


# =============================================================================
# 阶段2：提取关系
# =============================================================================

def extract_relationships(
    text: str,
    entities: List[Entity],
    session_time_iso: str = "",
) -> List[Relationship]:
    """
    阶段2：基于实体列表提取关系
    
    Args:
        text: 对话文本
        entities: 阶段1提取的实体列表
        session_time_iso: 会话时间
    
    Returns:
        关系列表
    """
    if not text or not text.strip():
        return []
    
    if not entities:
        return []
    
    if not _INSTRUCTOR_AVAILABLE:
        logger.warning("[consolidated_extractor] instructor not available")
        return []
    
    client, model = _get_instructor_client()
    if not client:
        return []
    
    # 构建实体列表文本
    entities_text = "\n".join([
        f"- {e.name} ({e.type}): {e.description}"
        for e in entities
    ])
    
    prompt = RELATIONSHIP_EXTRACTION_PROMPT.format(
        entities_text=entities_text,
        text=text,
        session_time=session_time_iso or "unknown",
    )
    
    try:
        result = client.chat.completions.create(
            model=model,
            response_model=RelationshipExtractionResult,
            messages=[{"role": "user", "content": prompt}],
            max_retries=2,
        )
        
        relationships = result.relationships or []
        
        # 验证：source 和 target 必须在实体列表中
        entity_names = {e.name for e in entities}
        entity_names_lower = {e.name.lower() for e in entities}
        
        validated = []
        for rel in relationships:
            source_ok = rel.source in entity_names or rel.source.lower() in entity_names_lower
            target_ok = rel.target in entity_names or rel.target.lower() in entity_names_lower
            
            if source_ok and target_ok:
                validated.append(rel)
            elif _DEBUG:
                logger.warning(
                    f"[consolidated_extractor] skipping invalid relationship: "
                    f"{rel.source} -[{rel.type}]-> {rel.target}"
                )
        
        if _DEBUG:
            logger.info(f"[consolidated_extractor] extracted {len(validated)} relationships")
            for r in validated:
                logger.info(f"  - {r.source} -[{r.type}]-> {r.target}")
        
        return validated
    
    except Exception as e:
        logger.error(f"[consolidated_extractor] relationship extraction failed: {e}")
        return []


# =============================================================================
# 阶段3：LLM 验证+补漏
# =============================================================================

def validate_and_补漏(
    text: str,
    entities: List[Entity],
    relationships: List[Relationship],
    session_time_iso: str = "",
) -> tuple[List[Entity], List[Relationship]]:
    """
    阶段3：LLM 验证+补漏
    
    检查提取结果是否完整，补充遗漏的实体和关系
    
    Args:
        text: 原始文本
        entities: 已提取的实体
        relationships: 已提取的关系
        session_time_iso: 会话时间
    
    Returns:
        (补充后的实体列表, 补充后的关系列表)
    """
    if not _ENABLE_LLM_VALIDATION:
        return entities, relationships
    
    if not _INSTRUCTOR_AVAILABLE:
        return entities, relationships
    
    client, model = _get_instructor_client()
    if not client:
        return entities, relationships
    
    # 构建已提取内容的文本
    entities_text = "\n".join([
        f"- {e.name} ({e.type})"
        for e in entities
    ]) or "(none)"
    
    relationships_text = "\n".join([
        f"- {r.source} -[{r.type}]-> {r.target}"
        for r in relationships
    ]) or "(none)"
    
    prompt = VALIDATION_PROMPT.format(
        text=text,
        entities_text=entities_text,
        relationships_text=relationships_text,
        session_time=session_time_iso or "unknown",
    )
    
    try:
        result = client.chat.completions.create(
            model=model,
            response_model=ValidationResult,
            messages=[{"role": "user", "content": prompt}],
            max_retries=2,
        )
        
        # 合并实体
        entity_names = {e.name.lower() for e in entities}
        new_entities = list(entities)
        
        # 添加遗漏的实体
        for e in (result.missing_entities or []):
            if e.name.lower() not in entity_names:
                new_entities.append(e)
                entity_names.add(e.name.lower())
                if _DEBUG:
                    logger.info(f"[validation] added missing entity: {e.name} ({e.type})")
        
        # 修正实体类型
        for fix in (result.entities_to_fix or []):
            for i, e in enumerate(new_entities):
                if e.name.lower() == fix.name.lower():
                    if e.type != fix.type:
                        if _DEBUG:
                            logger.info(f"[validation] fixed entity type: {e.name} {e.type} -> {fix.type}")
                        new_entities[i] = Entity(
                            name=e.name,
                            type=fix.type,
                            description=e.description or fix.description,
                        )
                    break
        
        # 合并关系
        rel_keys = {(r.source.lower(), r.type, r.target.lower()) for r in relationships}
        new_relationships = list(relationships)
        
        # 添加遗漏的关系
        all_entity_names = {e.name.lower() for e in new_entities}
        for r in (result.missing_relationships or []):
            key = (r.source.lower(), r.type, r.target.lower())
            if key not in rel_keys:
                # 验证 source 和 target 存在
                if r.source.lower() in all_entity_names and r.target.lower() in all_entity_names:
                    new_relationships.append(r)
                    rel_keys.add(key)
                    if _DEBUG:
                        logger.info(f"[validation] added missing relationship: {r.source} -[{r.type}]-> {r.target}")
        
        if _DEBUG:
            added_entities = len(new_entities) - len(entities)
            added_rels = len(new_relationships) - len(relationships)
            if added_entities > 0 or added_rels > 0:
                logger.info(f"[validation] added {added_entities} entities, {added_rels} relationships")
        
        return new_entities, new_relationships
    
    except Exception as e:
        logger.warning(f"[validation] failed: {e}")
        return entities, relationships


# =============================================================================
# 辅助函数：将关系转换为自然语言 facts
# =============================================================================

def _generate_facts_from_relationships(
    relationships: List[Relationship],
    entities: List[Entity],
    session_time_iso: str = "",
) -> List[str]:
    """
    将结构化关系转换为自然语言 facts
    
    Args:
        relationships: 关系列表
        entities: 实体列表（用于获取实体类型）
        session_time_iso: 会话时间
    
    Returns:
        自然语言 facts 列表
    """
    if not relationships:
        return []
    
    # 构建实体名称到类型的映射
    entity_type_map = {e.name.lower(): e.type for e in entities}
    
    facts = []
    
    for rel in relationships:
        try:
            source = rel.source
            target = rel.target
            rel_type = rel.type
            
            # 获取实体类型
            source_type = entity_type_map.get(source.lower(), "Entity")
            target_type = entity_type_map.get(target.lower(), "Entity")
            
            # 根据关系类型生成自然语言
            fact = _relationship_to_natural_language(
                source=source,
                target=target,
                rel_type=rel_type,
                source_type=source_type,
                target_type=target_type,
                description=rel.description,
            )
            
            if fact:
                # 添加时间信息（如果有）
                if session_time_iso:
                    fact = f"{fact} (recorded: {session_time_iso})"
                facts.append(fact)
        
        except Exception as e:
            # 失败时使用结构化格式作为 fallback
            fallback = f"({rel.source}) -[{rel.type}]-> ({rel.target})"
            facts.append(fallback)
            if _DEBUG:
                logger.warning(f"[_generate_facts] failed to convert: {fallback}, error: {e}")
    
    return facts


def _relationship_to_natural_language(
    source: str,
    target: str,
    rel_type: str,
    source_type: str = "Entity",
    target_type: str = "Entity",
    description: str = "",
) -> str:
    """
    将单个关系转换为自然语言句子
    
    使用通用规则，不依赖特定关系类型的硬编码
    
    Args:
        source: 源实体名称
        target: 目标实体名称
        rel_type: 关系类型（大写，下划线分隔）
        source_type: 源实体类型
        target_type: 目标实体类型
        description: 关系描述
    
    Returns:
        自然语言句子
    """
    # 特殊关系类型的模板
    special_templates = {
        # 状态变化
        "WAS": "{source} was {target} (past state)",
        "USED_TO_BE": "{source} used to be {target} (past state)",
        "IS": "{source} is {target}",
        "BECAME": "{source} became {target}",
        
        # 访问/参与
        "VISITED": "{source} visited {target}",
        "ATTENDED": "{source} attended {target}",
        "WENT_TO": "{source} went to {target}",
        
        # 所有权
        "OWNS": "{source} owns {target}",
        "HAS": "{source} has {target}",
        "OWNS_COUNT": "{source} owns {target}",
        "HAS_COUNT": "{source} has {target}",
        
        # 工作/居住
        "WORKS_AT": "{source} works at {target}",
        "LIVES_IN": "{source} lives in {target}",
        "LOCATED_AT": "{source} is located at {target}",
        "LOCATED_IN": "{source} is located in {target}",
        
        # 关系
        "KNOWS": "{source} knows {target}",
        "FRIEND_OF": "{source} is a friend of {target}",
        "COLLEAGUE_OF": "{source} is a colleague of {target}",
        
        # 偏好
        "PREFERS": "{source} prefers {target}",
        "LIKES": "{source} likes {target}",
        "LOVES": "{source} loves {target}",
        "FAVORITE": "{source}'s favorite is {target}",
        
        # 学习/兴趣
        "INTERESTED_IN": "{source} is interested in {target}",
        "STUDYING": "{source} is studying {target}",
        "LEARNING": "{source} is learning {target}",
        "WANTS_TO_LEARN": "{source} wants to learn {target}",
        
        # 计划
        "PLANS_TO_VISIT": "{source} plans to visit {target}",
        "PLANNING": "{source} is planning {target}",
        
        # 购买/消费
        "PURCHASED": "{source} purchased {target}",
        "BOUGHT": "{source} bought {target}",
        "PURCHASED_FROM": "{source} purchased from {target}",
        
        # 预约/会面
        "HAD_APPOINTMENT_WITH": "{source} had an appointment with {target}",
        "MET_WITH": "{source} met with {target}",
        
        # 数值关系
        "COST": "{source} cost {target}",
        "PRICE": "{source} price is {target}",
        "DURATION": "{source} duration is {target}",
        "FREQUENCY": "{source} frequency is {target}",
        "DILUTION_RATIO": "{source} dilution ratio is {target}",
        "MIX_RATIO": "{source} mix ratio is {target}",
        "DISCOUNT_PERCENT": "{source} discount is {target}",
        "FIRST_ORDER_DISCOUNT_PERCENT": "{source} first order discount is {target}",
        "WAIT_TIME": "{source} wait time was {target}",
        
        # 演讲/表演
        "SPOKE_AT": "{source} spoke at {target}",
        "PERFORMED_AT": "{source} performed at {target}",
    }
    
    # 检查是否有特殊模板
    rel_type_upper = rel_type.upper()
    if rel_type_upper in special_templates:
        template = special_templates[rel_type_upper]
        return template.format(source=source, target=target)
    
    # 通用转换：将下划线替换为空格，转小写
    # 例如：FIRST_ORDER_DISCOUNT -> "first order discount"
    rel_phrase = rel_type.lower().replace("_", " ")
    
    # 根据关系类型的语义选择句式
    # 动词类关系（通常以动词开头）
    verb_prefixes = ["has", "had", "is", "was", "did", "does", "can", "will", "would", "should"]
    
    if any(rel_phrase.startswith(prefix) for prefix in verb_prefixes):
        # 关系本身是动词短语
        return f"{source} {rel_phrase} {target}"
    else:
        # 默认：source [relation] target
        return f"{source} {rel_phrase} {target}"


# =============================================================================
# 主函数：三阶段提取（实体 → 关系 → 验证补漏）
# =============================================================================

def extract_knowledge_graph(
    text: str,
    session_time_iso: str = "",
    turn_id: int = 0,
) -> Dict[str, Any]:
    """
    多阶段提取：简单事实 → 实体 → 关系 → 验证补漏
    
    Args:
        text: 对话文本
        session_time_iso: 会话时间
        turn_id: 轮次 ID
    
    Returns:
        {
            "simple_facts": List[str],  # 简单事实（精炼信息）
            "entities": List[Entity],
            "relationships": List[Relationship],
            "nodes": List[dict],  # 转换为 Neo4j 格式
            "relationships_neo4j": List[dict],  # 转换为 Neo4j 格式
        }
    """
    if not text or not text.strip():
        return {
            "simple_facts": [],
            "entities": [],
            "relationships": [],
            "nodes": [],
            "relationships_neo4j": [],
        }
    
    # 阶段0：提取简单事实（精炼信息，event_time 由 LLM 直接计算）
    simple_facts = extract_simple_facts(text, session_time_iso)
    
    # 阶段1：提取实体
    entities = extract_entities(text, session_time_iso)
    
    if not entities:
        # 如果没有实体，只返回简单事实
        return {
            "simple_facts": simple_facts,
            "entities": [],
            "relationships": [],
            "nodes": [],
            "relationships_neo4j": [],
        }
    
    # 阶段2：提取关系
    relationships = extract_relationships(text, entities, session_time_iso)
    
    # 阶段3：LLM 验证+补漏
    entities, relationships = validate_and_补漏(text, entities, relationships, session_time_iso)
    
    # 转换为 Neo4j 格式
    nodes = []
    for e in entities:
        nodes.append({
            "name": e.name,
            "label": e.type,
            "properties": {
                "description": e.description,
                "turn_id": turn_id,
            }
        })
    
    relationships_neo4j = []
    for r in relationships:
        # 查找 source 和 target 的类型
        source_type = "Person"
        target_type = "Organization"
        for e in entities:
            if e.name == r.source or e.name.lower() == r.source.lower():
                source_type = e.type
            if e.name == r.target or e.name.lower() == r.target.lower():
                target_type = e.type
        
        relationships_neo4j.append({
            "source_node_name": r.source,
            "source_node_label": source_type,
            "target_node_name": r.target,
            "target_node_label": target_type,
            "type": r.type,
            "properties": {
                "description": r.description,
                "strength": r.strength,
                "confidence": r.strength / 10.0,
                "turn_id": turn_id,
                "event_timestamp": session_time_iso,
            }
        })
    
    return {
        "simple_facts": simple_facts,
        "entities": entities,
        "relationships": relationships,
        "nodes": nodes,
        "relationships_neo4j": relationships_neo4j,
    }


# =============================================================================
# 转换为 KnowledgeGraphExtraction 格式（兼容现有代码）
# =============================================================================

def extract_to_kg_format(
    text: str,
    session_time_iso: str = "",
    turn_id: int = 0,
):
    """
    提取并转换为 KnowledgeGraphExtraction 格式
    
    兼容现有的 original_consolidation.py 流程
    """
    try:
        from memory.structured_memory import KnowledgeGraphExtraction, Node, Relationship as KGRelationship
    except ImportError:
        logger.error("[consolidated_extractor] cannot import KnowledgeGraphExtraction")
        return None
    
    result = extract_knowledge_graph(text, session_time_iso, turn_id)
    
    if not result["nodes"] and not result["simple_facts"]:
        return None
    
    # 转换 nodes
    kg_nodes = []
    for n in result["nodes"]:
        kg_nodes.append(Node(
            name=n["name"],
            label=n["label"],
            properties=n.get("properties", {}),
        ))
    
    # 转换 relationships
    kg_rels = []
    for r in result["relationships_neo4j"]:
        kg_rels.append(KGRelationship(
            source_node_name=r["source_node_name"],
            source_node_label=r["source_node_label"],
            target_node_name=r["target_node_name"],
            target_node_label=r["target_node_label"],
            type=r["type"],
            properties=r.get("properties", {}),
        ))
    
    # ✅ 传递简单事实（LLM 提取的精炼信息）
    # KnowledgeGraphExtraction.facts 期望 List[str]，所以需要编码为字符串
    # 格式：JSON 编码的字典，ltss_writer 会解析
    import json
    simple_facts = result.get("simple_facts", [])
    facts_for_kg = []
    for f in simple_facts:
        if isinstance(f, dict):
            # 新格式：{"text": "...", "source": "user/assistant", "event_time": "YYYY-MM-DD"}
            # 编码为 JSON 字符串，ltss_writer 会解析
            text = f.get("text", "")
            if text:
                fact_obj = {
                    "text": text,
                    "source": f.get("source", "user"),
                    "event_time": f.get("event_time", ""),
                }
                facts_for_kg.append(json.dumps(fact_obj, ensure_ascii=False))
        elif isinstance(f, str):
            facts_for_kg.append(f)
    
    return KnowledgeGraphExtraction(
        nodes=kg_nodes,
        relationships=kg_rels,
        facts=facts_for_kg,
        claims=[],
        insights=[],
    )



# =============================================================================
# 简化版巩固函数（替代 original_consolidation.py 的复杂流程）
# =============================================================================

def _chunk_text_by_turns(
    session_turns: List[Dict[str, Any]],
    include_assistant: bool,
    max_chars_per_chunk: int = 5000,
    overlap_turns: int = 1,
) -> List[str]:
    """
    将会话按轮次分块，确保每个 chunk 不超过 max_chars
    
    Args:
        session_turns: 会话轮次列表
        include_assistant: 是否包含助手回复
        max_chars_per_chunk: 每个 chunk 最大字符数
        overlap_turns: chunk 之间的重叠轮次数
    
    Returns:
        文本块列表
    """
    if not session_turns:
        return []
    
    # 构建每个 turn 的文本
    turn_texts = []
    for t in session_turns:
        role = (t.get("role") or "").strip().lower()
        if role not in ("user", "assistant"):
            continue
        if not include_assistant and role == "assistant":
            continue
        content = (t.get("content") or "").strip()
        if not content:
            continue
        turn_texts.append(f"{role.upper()}: {content}")
    
    if not turn_texts:
        return []
    
    # 如果总长度不超过 max_chars，直接返回
    full_text = "\n".join(turn_texts)
    if len(full_text) <= max_chars_per_chunk:
        return [full_text]
    
    # 分块
    chunks = []
    current_chunk = []
    current_len = 0
    
    for i, turn_text in enumerate(turn_texts):
        turn_len = len(turn_text) + 1  # +1 for newline
        
        if current_len + turn_len > max_chars_per_chunk and current_chunk:
            # 保存当前 chunk
            chunks.append("\n".join(current_chunk))
            
            # 开始新 chunk，包含重叠部分
            if overlap_turns > 0 and len(current_chunk) >= overlap_turns:
                current_chunk = current_chunk[-overlap_turns:]
                current_len = sum(len(t) + 1 for t in current_chunk)
            else:
                current_chunk = []
                current_len = 0
        
        current_chunk.append(turn_text)
        current_len += turn_len
    
    # 保存最后一个 chunk
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    
    return chunks


# Chunk 级别并行配置
# 注意：每个 chunk 会调用 4 次 LLM（简单事实、实体、关系、验证补漏）
# 设置较小的并发数，避免触发 API 速率限制
_CHUNK_PARALLEL_WORKERS = int(os.getenv("CHUNK_PARALLEL_WORKERS", "4"))


def consolidate_session_v2(
    *,
    session_turns: List[Dict[str, Any]],
    virtual_time: str,
    session_time_iso: str,
    include_assistant: bool = True,
    max_chars: int = 5000,
    chunk_parallel_workers: int = None,
) -> Optional[Dict[str, Any]]:
    """
    简化版巩固函数（两阶段提取）
    
    支持长文本分块处理，确保不会因为截断而丢失信息。
    支持 chunk 级别并行处理，加速提取。
    
    Args:
        session_turns: 会话轮次列表
        virtual_time: 虚拟时间（如 TURN_1）
        session_time_iso: ISO 格式时间戳
        include_assistant: 是否包含助手回复
        max_chars: 最大字符数（每个 chunk）
        chunk_parallel_workers: chunk 并行处理的 worker 数量（默认使用 _CHUNK_PARALLEL_WORKERS）
    
    Returns:
        {
            "nodes": List[dict],
            "relationships": List[dict],
            "kg_extraction": KnowledgeGraphExtraction,  # 兼容现有代码
        }
    """
    if not session_turns:
        return None
    
    # 解析 turn_id
    turn_id = 0
    try:
        import re
        m = re.search(r"(\d+)", str(virtual_time or ""))
        turn_id = int(m.group(1)) if m else 0
    except Exception:
        pass
    
    # 分块处理长文本
    chunks = _chunk_text_by_turns(
        session_turns,
        include_assistant=include_assistant,
        max_chars_per_chunk=max_chars,
        overlap_turns=1,
    )
    
    if not chunks:
        return None
    
    if _DEBUG:
        logger.info(f"[consolidate_v2] turn_id={turn_id} chunks={len(chunks)}")
        for i, chunk in enumerate(chunks):
            logger.info(f"[consolidate_v2] chunk[{i}] len={len(chunk)}")
    
    # 确定并行 worker 数量
    workers = chunk_parallel_workers if chunk_parallel_workers is not None else _CHUNK_PARALLEL_WORKERS
    workers = max(1, min(workers, len(chunks)))  # 不超过 chunk 数量
    
    # 定义单个 chunk 的处理函数
    def process_single_chunk(chunk_idx: int, chunk_text: str):
        """处理单个 chunk，返回提取结果"""
        if _DEBUG:
            logger.info(f"[consolidate_v2] processing chunk {chunk_idx+1}/{len(chunks)}")
        
        try:
            kg_extraction = extract_to_kg_format(chunk_text, session_time_iso, turn_id)
            return (chunk_idx, kg_extraction)
        except Exception as e:
            logger.error(f"[consolidate_v2] chunk {chunk_idx} extraction failed: {e}")
            return (chunk_idx, None)
    
    # 并行处理所有 chunks
    chunk_results = []
    
    if workers > 1 and len(chunks) > 1:
        # 并行处理
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        if _DEBUG:
            logger.info(f"[consolidate_v2] parallel processing with {workers} workers")
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_single_chunk, i, chunk_text): i
                for i, chunk_text in enumerate(chunks)
            }
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    chunk_results.append(result)
                except Exception as e:
                    chunk_idx = futures[future]
                    logger.error(f"[consolidate_v2] chunk {chunk_idx} worker failed: {e}")
    else:
        # 串行处理（单个 chunk 或 workers=1）
        for i, chunk_text in enumerate(chunks):
            result = process_single_chunk(i, chunk_text)
            chunk_results.append(result)
    
    # 按 chunk 顺序排序结果
    chunk_results.sort(key=lambda x: x[0])
    
    # 合并所有 chunk 的结果
    all_entities = []
    all_relationships = []
    all_facts = []  # ✅ 收集所有 facts
    entity_names_seen = set()
    rel_keys_seen = set()
    
    for chunk_idx, kg_extraction in chunk_results:
        if not kg_extraction:
            continue
        
        # 合并实体（去重）
        for n in (kg_extraction.nodes or []):
            name_lower = n.name.lower()
            if name_lower not in entity_names_seen:
                entity_names_seen.add(name_lower)
                all_entities.append({
                    "name": n.name,
                    "label": n.label,
                    "properties": n.properties if isinstance(n.properties, dict) else {},
                })
        
        # 合并关系（去重）
        for r in (kg_extraction.relationships or []):
            key = (
                r.source_node_name.lower(),
                r.type.upper(),
                r.target_node_name.lower(),
            )
            if key not in rel_keys_seen:
                rel_keys_seen.add(key)
                all_relationships.append({
                    "source_node_name": r.source_node_name,
                    "source_node_label": r.source_node_label,
                    "target_node_name": r.target_node_name,
                    "target_node_label": r.target_node_label,
                    "type": r.type,
                    "properties": r.properties if isinstance(r.properties, dict) else {},
                })
        
        # ✅ 收集 facts（去重）
        for fact in (kg_extraction.facts or []):
            if isinstance(fact, str) and fact.strip() and fact not in all_facts:
                all_facts.append(fact)
    
    if not all_entities:
        return None
    
    if _DEBUG:
        logger.info(f"[consolidate_v2] merged: nodes={len(all_entities)} relationships={len(all_relationships)} facts={len(all_facts)}")
    
    # 构建 KnowledgeGraphExtraction
    try:
        from memory.structured_memory import KnowledgeGraphExtraction, Node, Relationship as KGRelationship
        
        kg_nodes = [
            Node(name=n["name"], label=n["label"], properties=n.get("properties", {}))
            for n in all_entities
        ]
        kg_rels = [
            KGRelationship(
                source_node_name=r["source_node_name"],
                source_node_label=r["source_node_label"],
                target_node_name=r["target_node_name"],
                target_node_label=r["target_node_label"],
                type=r["type"],
                properties=r.get("properties", {}),
            )
            for r in all_relationships
        ]
        
        kg_extraction = KnowledgeGraphExtraction(
            nodes=kg_nodes,
            relationships=kg_rels,
            facts=all_facts,  # ✅ 传递收集的 facts
            claims=[],
            insights=[],
        )
    except ImportError:
        kg_extraction = None
    
    return {
        "nodes": all_entities,
        "relationships": all_relationships,
        "kg_extraction": kg_extraction,
    }
