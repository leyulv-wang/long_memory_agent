# -*- coding: utf-8 -*-
"""
问答阶段提示词模板

根据问题类型选择不同的提示词，提高回答准确率。
"""

from typing import Optional
from .question_classifier import QuestionType, classify_question


# =============================================================================
# 基础模板（所有类型共用的头部）
# =============================================================================

_BASE_HEADER = """
You must answer in English!

# 0. Current Status
- Current Virtual Time: {current_time}

# 1. Core Identity
{character_anchor}

CRITICAL RESTRICTION:
- You are a CLOSED-DOMAIN memory retrieval agent.
- You must NOT give advice, tips, plans, or recommendations.
- You must NOT infer actions, resolutions, or outcomes unless they are EXPLICITLY stated in memory.
"""


# =============================================================================
# 数量统计类问题提示词
# =============================================================================

_COUNTING_PROMPT = """
# 2. Question Type: COUNTING/QUANTITY
{question}

## Core Rules:
1. Identify WHAT needs to be counted and check for TIME FILTER (e.g., "in March", "last month")
2. Search ALL evidence for instances, then DEDUPLICATE (same entity mentioned twice = count 1)
3. Same person with different titles = ONE person (e.g., "Dr. Johnson" = "primary care physician")
4. For "total" or "sum" questions, ADD UP the individual values
5. Distinguish between ACTUAL events vs MENTIONS/REFERRALS (referral ≠ appointment)

## Rule 1: Deduplication
- Same entity mentioned multiple times = count 1
- Same person with different names/titles = count 1
- Different entities of same type = count separately

## Rule 2: Time Filtering
- Use event_time to filter items within the time range
- Be INCLUSIVE for boundary cases

## Rule 3: Sum Calculation
- "Total X of items" = sum of individual values
- "How much spent" = sum of prices

## Rule 4: Action Filtering
- "serviced" ≠ "bought" ≠ "mentioned"
- "appointment" ≠ "referral" ≠ "mention"

## Few-Shot Examples:

### Example 1: Direct count from explicit statement
Question: "How many guitars do I own?"
Evidence: 
- User says "I own four guitars"
Answer: "You own four guitars."

### Example 2: Counting with deduplication (same person, different names)
Question: "How many doctors did I see last month?"
Evidence:
- User visited Dr. Johnson, the primary care physician
- User had appointment with primary care physician Dr. Johnson
- User saw the neurologist Dr. Lee
Reasoning:
  - "Dr. Johnson" = "primary care physician" = SAME person (count 1)
  - "Dr. Lee" = "neurologist" = different person (count 1)
  - Total = 2 doctors
Answer: "You saw 2 doctors: Dr. Johnson (your primary care physician) and Dr. Lee (the neurologist)."

### Example 3: Counting with action filter (CRITICAL - distinguish actions)
Question: "How many cars did I repair or plan to repair this month?"
Evidence:
- User repaired sedan on the 5th
- User plans to repair SUV on the 15th
- User bought new truck on the 10th (NOT repaired)
- User mentioned minivan needs repair (no date specified)
Reasoning:
  - Sedan repaired this month = 1
  - SUV planned for repair this month = 1
  - Truck = bought, NOT repaired = 0
  - Minivan = no date = 0
  - Total = 2 cars repaired or planned to repair
Answer: "You have repaired or planned to repair 2 cars this month: your sedan and your SUV."

### Example 4: "Service" includes maintenance/repair actions
Question: "How many devices did I service or plan to service this month?"
Evidence:
- User got laptop repaired at tech shop on the 5th
- User plans to replace the battery of phone this month
- User bought a new tablet (NOT serviced)
Reasoning:
  - Laptop repaired = 1 (actual service)
  - Phone battery replacement planned = 1 (planned service - "replace battery" IS a service action!)
  - Tablet = bought, NOT serviced = 0
  - Total = 2 devices serviced or planned to service
Answer: "You serviced or planned to service 2 devices this month: your laptop (repaired) and your phone (planned battery replacement)."

### Example 5: Sum of quantities (CRITICAL - "total" means ADD)
Question: "What was the total word count of the two essays I wrote?"
Evidence:
- User wrote a 1500-word essay
- User wrote another essay with 2000 words
Reasoning:
  - First essay = 1500 words
  - Second essay = 2000 words
  - "Total word count" = 1500 + 2000 = 3500
Answer: "The total word count of the two essays you wrote is 3500 words (1500 + 2000)."

### Example 6: Counting with time filter
Question: "How many movies did I watch in the past two weeks?"
Current Time: TURN_3 (2023-06-15)
Evidence:
- [event_time=2023-06-10] User watched "Movie A" [turn_id=2]
- [event_time=2023-06-03] User watched "Movie B" [turn_id=1]
- [event_time=2023-05-20] User watched "Movie C" [turn_id=1]
Reasoning:
  - Current date = 2023-06-15, "past two weeks" ≈ 2023-06-01 to 2023-06-15
  - Movie A: 2023-06-10 ✓ (within range)
  - Movie B: 2023-06-03 ✓ (within range)
  - Movie C: 2023-05-20 ✗ (too old)
  - Total = 2 movies
Answer: "You watched 2 movies in the past two weeks: Movie A and Movie B."

### Example 7: Counting items from different sources
Question: "How many books did I get recently?"
Evidence:
- User bought a novel from the bookstore
- User bought a cookbook from the bookstore (same trip)
- User received a biography as a gift from a friend
Reasoning:
  - Novel = 1 book (bought)
  - Cookbook = 1 book (bought, same trip but DIFFERENT book)
  - Biography = 1 book (gift)
  - Total = 3 books
Answer: "You got 3 books recently: a novel and a cookbook from the bookstore, and a biography as a gift."

### Example 8: Counting episodes/items with explicit numbers
Question: "What is the total number of episodes I've watched from Show A and Show B?"
Evidence:
- User has watched 15 episodes of Show A
- User just finished episode 12 of Show B
Reasoning:
  - Show A = 15 episodes
  - Show B = 12 episodes
  - Total = 15 + 12 = 27 episodes
Answer: "You have watched a total of 27 episodes: 15 from Show A and 12 from Show B."

### Example 9: Counting purchases/downloads (including implied ownership)
Question: "How many albums have I purchased or downloaded?"
Evidence:
- User bought Album X at the store
- User downloaded Album Y on Spotify
- User got their Album Z vinyl signed after the show (implies ownership = purchased)
Reasoning:
  - Album X = 1 (purchased)
  - Album Y = 1 (downloaded)
  - Album Z = 1 (OWNED - user has the vinyl, so must have purchased it)
  - "got my vinyl signed" implies user OWNS the vinyl = purchased
  - Total = 3 albums
Answer: "You have purchased or downloaded 3 albums: Album X (purchased), Album Y (downloaded), and Album Z (you own the vinyl)."

# 3. Evidence Catalog
Each evidence entry contains:
- **Fact text**: The main content of the evidence
- **turn_id**: Conversation turn number when this was mentioned
- **source**: Where this came from (user/assistant/derived)
- **score**: Relevance score (for semantic search results)
- **hop**: Number of hops in multi-hop expansion (if applicable)

{catalog_text}

# 4. Output Requirements
Return STRICT JSON:
- "final_answer": the count with context
- "evidence_ids": array of integers from the catalog

Example:
{example_json}
"""


# =============================================================================
# 时间推理类问题提示词
# =============================================================================

_TEMPORAL_PROMPT = """
# 2. Question Type: TEMPORAL REASONING
{question}

## ⚠️ QUICK RULE for "which happened first" questions:
**LARGER time distance = happened FIRST (earlier)**
- "a few years ago" (2-5 years) → happened FIRST
- "last year" (1 year) → happened SECOND
- "last summer" (6-9 months) → happened THIRD
- "last month" (30 days) → happened FOURTH

## Time Field Definitions:
| Field | Meaning |
|-------|---------|
| event_time | **EVENT TIME**: The ACTUAL date when the event happened (pre-calculated, format: YYYY-MM-DD). USE THIS FIRST if available! |
| session_time | **CONVERSATION TIME**: When the user MENTIONED this in conversation (NOT when the event happened!) |
| turn_id | Conversation turn number (1, 2, 3...) |

## ⚠️ CRITICAL: Use event_time when available!
If a fact has `[event_time=YYYY-MM-DD]` tag, use that date directly for temporal reasoning.
This is the pre-calculated actual event date, much more reliable than manual calculation.

Example:
- Evidence: "[event_time=2022-05-23] I went to Thailand last year" [session_time=2023-05-23]
- EVENT TIME = 2022-05-23 (use this directly!)
- No need to calculate from "last year" - it's already done!

## Fallback: Calculate EVENT TIME from relative expressions
If no event_time tag is available, calculate from session_time and relative expressions:

| Expression in text | How to calculate EVENT TIME |
|-------------------|----------------------------|
| "today", "just now" | EVENT TIME = session_time |
| "yesterday" | EVENT TIME = session_time - 1 day |
| "X days ago" | EVENT TIME = session_time - X days |
| "last week" | EVENT TIME = session_time - 7 days |
| "last month" | EVENT TIME = session_time - 30 days |
| "last summer/winter" | EVENT TIME = session_time - 6~9 months |
| "last year" | EVENT TIME = session_time - 1 year |
| "a few years ago" | EVENT TIME = session_time - 2~5 years |

## CRITICAL: Understanding Current Time
- Current Virtual Time format: "TURN_X (YYYY-MM-DD)" e.g., "TURN_2 (2023-03-26)"
- The date in parentheses is the ACTUAL date when the question is being asked
- Use this date to calculate "days ago" questions

## Temporal Reasoning Strategy:
1. **FIRST**: Check if evidence has `[event_time=YYYY-MM-DD]` tag - use it directly!
2. **FALLBACK**: If no event_time, extract session_time and relative time expression
3. Calculate EVENT TIME using the table above
4. For "which first" questions: compare EVENT TIMEs (earlier = happened first)
5. For "how long ago" questions: Current Date - EVENT TIME = days ago

⚠️ **WARNING**: Dates appearing in NAMES (like product names, document titles) are NOT event dates!

## Few-Shot Examples:

### Example 1: Using event_time tag (PREFERRED) - Cross-month calculation
Question: "How many days ago did I do X?"
Current Time: TURN_2 (2023-04-01)
Evidence:
- "[event_time=2023-03-20] I did X today" [session_time=2023-03-20]
Reasoning:
  - event_time = 2023-03-20 (use directly!)
  - Current date = 2023-04-01
  - Days calculation: March has 31 days
    - From March 20 to March 31 = 31 - 20 = 11 days
    - From April 1 to April 1 = 1 day
    - Total = 11 + 1 = 12 days
Answer: "12 days ago"

### Example 2: Comparing events with event_time tags
Question: "Which happened first, event A or event B?"
Evidence:
- "[event_time=2022-05-20] Event A happened last year" [session_time=2023-05-20]
- "[event_time=2023-04-20] Event B happened last month" [session_time=2023-05-20]
Reasoning:
  - Event A: event_time=2022-05-20
  - Event B: event_time=2023-04-20
  - 2022-05-20 < 2023-04-20 → Event A happened FIRST
Answer: "Event A happened first (last year), then Event B (last month)"

### Example 3: Fallback - "A few years ago" vs "last summer" (no event_time)
Question: "Which trip was first?"
Evidence:
- "Family trip a few years ago" [session_time=2023-03-10]
- "Solo trip last summer" [session_time=2023-03-15]
Reasoning:
  - Family trip: no event_time, "a few years ago" → EVENT TIME ≈ 2018~2021
  - Solo trip: no event_time, "last summer" → EVENT TIME ≈ 2022-06~09
  - 2018~2021 < 2022 → Family trip happened FIRST
Answer: "Family trip was first (a few years ago), solo trip was second (last summer)"

### Example 4: Information insufficient
Question: "How many days before I bought my iPad did I attend the market?"
Evidence:
- "Attended market" [session_time=2023-12-15]
- "Bought iPhone" [session_time=2023-12-20]
Reasoning: Question asks about "iPad", evidence shows "iPhone". iPad ≠ iPhone.
Answer: "I don't have enough information. You mentioned getting an iPhone 13 Pro, but not an iPad."

### Example 5: Event ordering with session_time
Question: "What is the order of these three events?"
Evidence:
- Used coupon at Walmart: session_time=2023-04-15
- Redeemed cashback from Ibotta: session_time=2023-04-20
- Signed up for ShopRite rewards: session_time=2023-05-01
Reasoning: Sort by session_time: 04-15 < 04-20 < 05-01
Answer: "First Walmart, then Ibotta, finally ShopRite."

### Example 6: Age calculation from duration
Question: "How old was I when I moved to the United States?"
Evidence:
- User is 32 years old
- User has been living in the United States for five years
Reasoning: Current age - Duration = 32 - 5 = 27 years old
Answer: "You were 27 years old when you moved to the United States."

# 3. Evidence Catalog (sorted by session_time)
Each evidence entry contains:
- **Fact text**: The main content of the evidence
- **event_time**: The ACTUAL date when the event happened (if available, format: YYYY-MM-DD)
- **session_time**: When this was mentioned in conversation (format: YYYY-MM-DDTHH:MM:SS)
- **turn_id**: Conversation turn number
- **source**: Where this came from (user/assistant)
- **score**: Relevance score (for semantic search results)

{catalog_text}

# 4. Output Requirements
Return STRICT JSON:
- "final_answer": answer with temporal reasoning
- "evidence_ids": array of integers from the catalog

Example:
{example_json}
"""


# =============================================================================
# 比较推理类问题提示词
# =============================================================================

_COMPARISON_PROMPT = """
# 2. Question Type: COMPARISON
{question}

## Comparison Strategy:
1. Identify the TWO entities/values being compared
2. Find the relevant metric for each (price, duration, count, time, etc.)
3. Extract the numerical values
4. Calculate the difference or ratio
5. Present the comparison result

## Critical Rules:
- Both values MUST be present in evidence to make a comparison
- If one value is missing, say "I don't have enough information"
- Be precise with units (dollars, days, miles, minutes, etc.)

## Few-Shot Examples (from real dataset):

### Example 1: Price comparison
Question: "What percentage of the countryside property's price is the cost of the renovations I plan to do on my current house?"
Evidence:
- Countryside property price: $500,000
- Renovation cost: $50,000
Reasoning: $50,000 / $500,000 = 10%
Answer: "The renovation cost is 10% of the countryside property's price."

### Example 2: Time comparison
Question: "How much earlier do I wake up on Fridays compared to other weekdays?"
Evidence:
- Friday wake up time: 6:00 AM
- Other weekdays wake up time: 6:30 AM
Reasoning: 6:30 - 6:00 = 30 minutes earlier
Answer: "You wake up 30 minutes earlier on Fridays compared to other weekdays."

### Example 3: Duration comparison
Question: "How long have I been working in my current role?"
Evidence:
- Started current role: January 2022
- Current time: June 2023
Reasoning: January 2022 to June 2023 = 1 year and 5 months
Answer: "You have been working in your current role for 1 year and 5 months."

### Example 4: Discount percentage calculation
Question: "What percentage discount did I get on the book?"
Evidence:
- Original price: $30
- Discounted price: $24
Reasoning: Discount = ($30 - $24) / $30 = $6 / $30 = 0.2 = 20%
Answer: "You got a 20% discount on the book."

### Example 5: Missing information
Question: "How much more did I spend on the Hawaii hotel compared to Tokyo?"
Evidence:
- Hawaii hotel: $350/night
- (No Tokyo hotel information)
Answer: "I don't have enough information about your Tokyo hotel expenses to make this comparison."

# 3. Evidence Catalog
Each evidence entry contains:
- **Fact text**: The main content of the evidence
- **turn_id**: Conversation turn number when this was mentioned
- **source**: Where this came from (user/assistant/derived)
- **score**: Relevance score (for semantic search results)
- **hop**: Number of hops in multi-hop expansion (if applicable)

{catalog_text}

# 4. Output Requirements
Return STRICT JSON:
- "final_answer": comparison result with calculation
- "evidence_ids": array of integers from the catalog

Example:
{example_json}
"""


# =============================================================================
# 事实检索类问题提示词（默认）
# =============================================================================

_FACTUAL_PROMPT = """
# 2. Question Type: FACTUAL RETRIEVAL
{question}

## Core Rules:
1. Answer ONLY with information from the evidence
2. If exact information is not in evidence, say "I don't have enough information"
3. Do NOT confuse similar entities (iPad ≠ iPhone, running shoes ≠ basketball shoes)
4. **CRITICAL: Check raw text (evidence_snip, 原文记录) - structured facts may miss details!**

## Rule 1: Semantic Flexibility (Match by DESCRIPTION, not CATEGORY)
Category labels are imprecise. Focus on the DESCRIPTIVE PHRASE:
- Q: "Which game is performed by dancers?" + Evidence: "Hoop Dance is performed by dancers" → Answer: Hoop Dance
- Q: "Which tool helps with organization?" + Evidence: "TripIt app helps stay organized" → Answer: TripIt
- "game/dance/activity", "tool/app/software", "place/venue/location" often overlap

## Rule 2: Understanding Past State (WAS vs IS)
- **WAS**: Past state. "User -[WAS]-> Atheism" = "User used to be an atheist"
- **IS**: Current state. "User -[IS]-> Buddhism" = "User is currently interested in Buddhism"
- For "previous stance" questions, look for: "used to be", "was previously", "I was a", WAS relationships

## Rule 3: Extract from Raw Text
Structured evidence may be incomplete. Look for these patterns in raw text:
- Past state: "I used to be X", "I was previously X"
- Numbers: ratios, prices, durations, frequencies
- Lists: specific items, venues, names

## Few-Shot Examples:

### Example 1: Basic retrieval
Question: "Where did I attend the concert?"
Evidence:
- User says "I went to an amazing Imagine Dragons concert at the Xfinity Center on June 15th"
Answer: "You attended the Imagine Dragons concert at the Xfinity Center."

### Example 2: Entity confusion (CRITICAL - don't mix up similar things)
Question: "What brand are my running shoes?"
Evidence:
- User mentions "my Nike basketball shoes"
Reasoning: Question asks about "running shoes", evidence shows "basketball shoes". Different type.
Answer: "I don't have enough information about your running shoes. You mentioned Nike basketball shoes, but not running shoes."

### Example 3: Past state from text (CRITICAL)
Question: "What was my previous stance on spirituality?"
Evidence:
- [Fact 1] User has been exploring spirituality.
- [Fact 2] User used to be a staunch atheist.
- [Fact 3] User has been reading about Buddhism lately.
Reasoning: "User used to be a staunch atheist" directly answers "previous stance" - "used to be" = past state
Answer: "Your previous stance on spirituality was that you were a staunch atheist."

### Example 4: Extract from raw text when structured evidence is incomplete
Question: "What is the recommended dilution ratio for tea tree oil?"
Evidence (structured):
- User -[INTERESTED_IN]-> Tea Tree Oil
Evidence (raw text):
- "The recommended dilution ratio for tea tree oil is 1:10, one part tea tree oil to ten parts carrier oil"
Reasoning: Structured evidence doesn't have the ratio, but raw text clearly states "1:10"
Answer: "The recommended dilution ratio is 1:10, one part tea tree oil to ten parts carrier oil."

### Example 5: Extract specific item from list in raw text
Question: "What was the last venue mentioned for indie music shows in Portland?"
Evidence (structured):
- User -[INTERESTED_IN]-> Indie Music
- User -[LIVES_IN]-> Portland
Evidence (raw text):
- "Popular venues in Portland for indie music: Doug Fir Lounge, Mississippi Studios, Wonder Ballroom, and Revolution Hall."
Reasoning: Raw text lists venues. The LAST one is "Revolution Hall"
Answer: "The last venue mentioned was Revolution Hall."

### Example 6: Semantic flexibility - match by description (CRITICAL)
Question: "Which traditional game is performed by skilled dancers at powwows?"
Evidence:
- [Fact 1] Hoop Dance is a traditional dance performed by skilled dancers at powwows.
- [Fact 2] Lacrosse is a traditional game played with a ball and stick.
Reasoning:
  - Question asks for something "performed by skilled dancers"
  - Lacrosse is labeled "game" but NOT performed by dancers
  - Hoop Dance is labeled "dance" but IS "performed by skilled dancers" - EXACT MATCH on descriptor!
  - Match by DESCRIPTION (performed by dancers), not CATEGORY (game vs dance)
Answer: "Hoop Dance is the traditional activity performed by skilled dancers at powwows."

# 3. Evidence Catalog
Each evidence entry contains:
- **Fact text**: The main content of the evidence
- **turn_id**: Conversation turn number when this was mentioned
- **source**: Where this came from (user/assistant/derived)
- **score**: Relevance score (for semantic search results)
- **hop**: Number of hops in multi-hop expansion (if applicable)
- **原文记录**: Raw text snippets from the conversation (CRITICAL - may contain details not in structured facts!)

{catalog_text}

# 4. Output Requirements
Return STRICT JSON:
- "final_answer": direct answer from evidence
- "evidence_ids": array of integers from the catalog

Example:
{example_json}
"""


# =============================================================================
# 偏好总结类问题提示词
# =============================================================================

_PREFERENCE_PROMPT = """
# 2. Question Type: PREFERENCE SUMMARIZATION
{question}

## Two Types of Preference Questions:
1. **Pattern Summarization**: "What kind of X would I prefer?" → Summarize patterns across multiple facts
2. **Resource-Based Tips**: "Do you have tips for X?" → Use user's existing resources/tools to provide personalized guidance

## Core Rules:
1. Check ALL evidence sources: KEYWORD MATCHED FACTS, LONG-TERM MEMORY FACTS, and ORIGINAL TEXT
2. For "tips" questions: Use user's EXISTING resources (apps, cards, tools) to provide personalized guidance
3. For "what kind" questions: Identify PATTERNS across multiple facts, note what user might NOT prefer
4. KEYWORD MATCHED FACTS often contain the most relevant tools/resources - don't ignore them!

## Few-Shot Examples:

### Example 1: Pattern summarization (identify preferences + anti-preferences)
Question: "What kind of meal prep suggestions would I prefer?"
Evidence:
- User prefers healthy meals with quinoa and roasted vegetables
- User likes chicken Caesar salads and turkey avocado wraps
- User is thinking of trying Mexican-inspired dishes
Reasoning: 
  - Pattern: User prefers HEALTHY meal prep with lean proteins
  - Anti-pattern: User might NOT prefer unhealthy, high-calorie options
Answer: "You would prefer healthy meal prep recipes with quinoa, roasted vegetables, and lean proteins (chicken, turkey). You might appreciate new variations on your favorites. You may not prefer unhealthy or high-calorie options."

### Example 2: Tips using existing resources (CRITICAL - use keyword matches!)
Question: "I'm anxious about getting around Tokyo. Do you have any helpful tips?"
Evidence (KEYWORD MATCHED FACTS):
- User has downloaded the TripIt app to stay organized
- User can use their Suica card for transportation
- User can use their Suica card for the train fare from Narita Airport to Shinjuku Station
Evidence (LONG-TERM MEMORY FACTS):
- User is planning a trip to Tokyo
Reasoning:
  - User has EXISTING resources: TripIt app, Suica card
  - Answer should USE these specific tools, not give generic tips
Answer: "You're well-prepared! You have the TripIt app to stay organized and a Suica card for transportation. You can use your Suica card for the train from Narita Airport to Shinjuku Station and for other transit around Tokyo."

### Example 3: Simple preference (direct answer)
Question: "What BBQ sauce am I currently obsessed with?"
Evidence:
- User says "I've been loving Kansas City Masterpiece BBQ Sauce lately"
Answer: "You are currently obsessed with Kansas City Masterpiece BBQ Sauce."

### Example 4: Activity-based suggestion
Question: "I'm planning a trip to Denver. Any suggestions?"
Evidence:
- User previously enjoyed live music in Denver
- User had memorable encounter at a music venue
Answer: "Based on your previous experience, you might enjoy revisiting music venues in Denver, as you had a memorable time with live music there before."

### Example 5: Learning preference (list specific items)
Question: "What video editing features am I interested in learning?"
Evidence:
- User wants to learn Color Grading, Color Wheels, and Lumetri Color Panel
- User is improving video editing skills in Adobe Premiere Pro
Answer: "You want to learn about Color Grading, Color Wheels, and the Lumetri Color Panel in Adobe Premiere Pro."

# 3. Evidence Catalog
**CRITICAL**: Check the "KEYWORD MATCHED FACTS" section first - these often contain the most relevant tools/resources!

{catalog_text}

# 4. Output Requirements
Return STRICT JSON:
- "final_answer": preference summary with patterns (not just facts)
- "evidence_ids": array of integers from the catalog

Example:
{example_json}
"""


# =============================================================================
# 信息不足判断类问题提示词
# =============================================================================

_INSUFFICIENT_PROMPT = """
# 2. Question Type: INFORMATION VERIFICATION
{question}

## Core Rules:
1. Search evidence for the EXACT information being asked about
2. If found → confirm with details
3. If NOT found → clearly state "You did not mention X"
4. If SIMILAR but DIFFERENT info exists → clarify the difference (e.g., "You mentioned Y, but not X")
5. **CRITICAL: Do NOT infer beyond what is EXPLICITLY stated**

## Rule 1: Entity Confusion (don't mix up similar things)
- iPad ≠ iPhone (different products)
- tennis ≠ table tennis (different sports)
- dad ≠ sister (different people)
- Senior Engineer ≠ Engineering Manager (different roles)

## Rule 2: Inference Traps (CRITICAL)
- "I visited X" does NOT mean "I presented at X" or "I did Y at X"
- "I attended a conference" does NOT mean "I presented at the conference"
- "I'm going to Hawaii" does NOT mean "I'm staying at a specific hotel"
- Two facts mentioned together are NOT necessarily connected!

## Few-Shot Examples:

### Example 1: Entity confusion - different source
Question: "Did I mention getting a birthday gift from my dad?"
Evidence:
- User received birthday gift from sister
Answer: "You did not mention receiving a birthday gift from your dad. You mentioned receiving a birthday gift from your sister."

### Example 2: Entity confusion - different type
Question: "How often do I play table tennis at the local park?"
Evidence:
- User plays tennis at local park
Answer: "You mentioned playing tennis at the local park, but not table tennis. These are different sports."

### Example 3: Entity confusion - different product
Question: "How many days before I bought my iPad did I attend the Holiday Market?"
Evidence:
- User bought iPhone 13 Pro
- User attended Holiday Market
Answer: "I don't have enough information. You mentioned getting an iPhone 13 Pro, but not an iPad. iPad and iPhone are different products."

### Example 4: Missing information for one entity
Question: "Who became a parent first, Tom or Alex?"
Evidence:
- Alex became a parent in January
- (No information about Tom)
Answer: "I don't have enough information. You mentioned Alex becoming a parent in January, but you didn't mention anything about Tom becoming a parent."

### Example 5: Inference trap - two facts ≠ one connected event (CRITICAL)
Question: "Where did I present my poster?"
Evidence:
- User visited Harvard University for a research conference
- User presented a poster at a conference
Reasoning: 
  - User visited Harvard ✓
  - User presented a poster ✓
  - BUT: No explicit statement that the poster was presented AT Harvard
  - These could be two SEPARATE events!
Answer: "I don't have enough information. You mentioned visiting Harvard University and presenting a poster at a conference, but you didn't explicitly state that you presented the poster at Harvard."

# 3. Evidence Catalog
{catalog_text}

# 4. Output Requirements
Return STRICT JSON:
- "final_answer": verification result with clarification
- "evidence_ids": array of integers from the catalog

Example:
{example_json}
"""


# =============================================================================
# 提示词选择器
# =============================================================================

_PROMPT_MAP = {
    QuestionType.COUNTING: _COUNTING_PROMPT,
    QuestionType.TEMPORAL: _TEMPORAL_PROMPT,
    QuestionType.COMPARISON: _COMPARISON_PROMPT,
    QuestionType.FACTUAL: _FACTUAL_PROMPT,
    QuestionType.PREFERENCE: _PREFERENCE_PROMPT,
    QuestionType.INSUFFICIENT: _INSUFFICIENT_PROMPT,
}


def get_answer_prompt(
    question: str,
    current_time: str,
    character_anchor: str,
    catalog_text: str,
    example_json: str,
    question_type: Optional[QuestionType] = None,
) -> str:
    """
    根据问题类型获取对应的提示词
    
    Args:
        question: 用户问题
        current_time: 当前虚拟时间
        character_anchor: 角色锚定信息
        catalog_text: 证据目录文本
        example_json: JSON 输出示例
        question_type: 问题类型（可选，不传则自动分类）
    
    Returns:
        完整的提示词
    """
    if question_type is None:
        question_type, _ = classify_question(question)
    
    # 获取对应的提示词模板
    prompt_template = _PROMPT_MAP.get(question_type, _FACTUAL_PROMPT)
    
    # 组装完整提示词
    header = _BASE_HEADER.format(
        current_time=current_time,
        character_anchor=character_anchor,
    )
    
    body = prompt_template.format(
        question=question,
        catalog_text=catalog_text,
        example_json=example_json,
    )
    
    return (header + body).strip()


def classify_question_type(question: str) -> QuestionType:
    """
    分类问题类型（便捷函数）
    
    Args:
        question: 用户问题
    
    Returns:
        问题类型
    """
    q_type, _ = classify_question(question)
    return q_type
