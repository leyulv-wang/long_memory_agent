# -*- coding: utf-8 -*-
"""
巩固阶段提示词模板

用于从对话中提取知识图谱。
"""


def get_consolidation_prompt(
    session_text: str,
    recorded_turn_id: int,
    session_time_hint: str,
    chunk_index: int,
    chunk_total: int,
) -> str:
    """
    获取巩固阶段的提示词
    
    Args:
        session_text: 会话文本
        recorded_turn_id: 记录的 turn ID
        session_time_hint: 会话时间提示
        chunk_index: 当前 chunk 索引
        chunk_total: 总 chunk 数
    
    Returns:
        完整的提示词
    """
    return f"""
You are a knowledge graph extractor. Extract entities and relationships from the conversation.

=== CRITICAL: EXHAUSTIVE EXTRACTION ===
You MUST process EVERY USER turn in the conversation and extract ALL information from each turn.
DO NOT skip any turn. DO NOT summarize. Extract EVERYTHING mentioned by the user.

For EACH USER turn, extract:
1. ALL entities mentioned (people, places, organizations, objects, events)
2. ALL relationships between entities
3. ALL numerical values (counts, amounts, durations, percentages, prices)
4. ALL time expressions (dates, durations, frequencies)
5. ALL preferences, opinions, and facts stated by the user

OUTPUT FORMAT (strict JSON):
{{
  "nodes": [{{"name": str, "label": str, "properties": {{}}}}],
  "relationships": [{{"source_node_name": str, "source_node_label": str, "target_node_name": str, "target_node_label": str, "type": str, "properties": {{}}}}],
  "claims": [{{"text": str, "confidence": float, "knowledge_type": str, "source_of_belief": str}}],
  "insights": [str]
}}

NODE LABELS: Person, Organization, Location, Object, Event, Date, Value, Concept

ENTITY TYPE GUIDELINES:
- Organization: Physical places you can visit (museums, restaurants, galleries, theaters, hotels, shops, companies)
- Location: Geographic areas (cities, countries, parks, neighborhoods)
- Event: Time-bound activities (exhibitions, tours, lectures, conferences, meetings)
- Person: Individual people (including "User" for the speaker)
- Value: Numbers, counts, amounts, prices, percentages, durations (CRITICAL: always extract these!)
- Object: Physical items, products, possessions

RELATIONSHIP TYPE: UPPERCASE (e.g., VISITED, WORKS_AT, KNOWS, LIVES_IN, OWNS, HAS_COUNT, COSTS, LASTED)

=== KEY RULES ===

1. ENTITY NAMES must be proper nouns, NOT sentence fragments:
   - CORRECT: "Science Museum", "Dr. Rodriguez", "New York"
   - WRONG: "I visited the Science Museum"

2. VISITED target must be Organization/Location (the PLACE), not Event

3. CRITICAL: ALWAYS extract numerical values and quantities:
   - "I own three bikes" → User -[OWNS_COUNT]-> "3" (Value)
   - "I've read 5 issues" → User -[READ_COUNT]-> "5" (Value)
   - "I spent $75" → User -[SPENT]-> "$75" (Value)
   - "for three months" → User -[DURATION]-> "three months" (Value)
   - Numbers are CRITICAL for answering "how many" questions!

4. CRITICAL: ALWAYS extract time expressions:
   - "yesterday", "last week", "three months ago", "on June 15th"
   - Store in relationship properties: {{"time_expression": "yesterday"}}

5. Only create VISITED for places USER actually visited, not ASSISTANT recommendations

6. Time context: recorded_turn_id = {recorded_turn_id}, session_time = "{session_time_hint}"

7. Output limit: max 30 relationships, 40 nodes

=== EXAMPLE (Quantity extraction - CRITICAL) ===
Input: "I own three bikes and have been cycling for two years."
Output:
{{
  "nodes": [
    {{"name": "User", "label": "Person", "properties": {{}}}},
    {{"name": "Bikes", "label": "Object", "properties": {{}}}},
    {{"name": "3", "label": "Value", "properties": {{"type": "count"}}}},
    {{"name": "two years", "label": "Value", "properties": {{"type": "duration"}}}}
  ],
  "relationships": [
    {{"source_node_name": "User", "source_node_label": "Person", "target_node_name": "Bikes", "target_node_label": "Object", "type": "OWNS", "properties": {{"count": 3}}}},
    {{"source_node_name": "User", "source_node_label": "Person", "target_node_name": "3", "target_node_label": "Value", "type": "OWNS_COUNT", "properties": {{"entity": "bikes"}}}},
    {{"source_node_name": "User", "source_node_label": "Person", "target_node_name": "two years", "target_node_label": "Value", "type": "CYCLING_DURATION", "properties": {{}}}}
  ],
  "claims": [
    {{"text": "User owns 3 bikes", "confidence": 0.9, "knowledge_type": "observed_fact", "source_of_belief": "user_statement"}},
    {{"text": "User has been cycling for two years", "confidence": 0.9, "knowledge_type": "observed_fact", "source_of_belief": "user_statement"}}
  ],
  "insights": []
}}

=== SESSION TEXT (chunk {chunk_index}/{chunk_total}) ===
{session_text}
""".strip()
