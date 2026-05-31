"""
chains/langchain_chains.py
Две LangChain-цепочки:
  Chain 1 — Парсер сущностей: текст → JSON {поезд, путь, статус, ...}
  Chain 2 — Генератор инструкций: проблема + RAG контекст → текст-ответ для ДСП
"""

import json
import re
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage


# Системные промпты

ENTITY_PARSER_SYSTEM = """Ты — парсер сообщений дежурного по станции (ДСП) на железной дороге.
Извлеки из входного текста структурированные данные и верни ТОЛЬКО валидный JSON без пояснений.

Формат ответа (все поля обязательны, используй null если информация отсутствует):
{{
  "train_id": "номер поезда или null",
  "track": "номер пути или null",
  "status": "краткое описание статуса",
  "location": "место события или null",
  "issue": "описание проблемы или null",
  "action_required": true или false
}}

Примеры:
Вход: "Поезд 3002 стоит на пятом пути, заклинило колёсную пару"
Ответ: {{"train_id": "3002", "track": "5", "status": "задержка из-за неисправности", "location": "5 путь", "issue": "заклинивание колёсной пары", "action_required": true}}

Вход: "Поезд 45 прибыл на первый путь"
Ответ: {{"train_id": "45", "track": "1", "status": "прибытие", "location": "1 путь", "issue": null, "action_required": false}}"""

INSTRUCTION_GENERATOR_SYSTEM = """Ты — интеллектуальный помощник дежурного по станции (ДСП) на железной дороге России.

Твоя задача: на основе описания ситуации и выдержек из нормативных документов (если предоставлены)
сформулировать чёткие практические инструкции для ДСП.

Правила:
- Отвечай на русском языке, официальным техническим стилем
- Нумеруй шаги действий
- Если ситуация критическая (авария, сход, пожар) — начни ответ с "⛔ КРИТИЧЕСКАЯ СИТУАЦИЯ"
- Ссылайся на нормативный документ, если он предоставлен
- Будь конкретен: номер поезда, пути, действия
- В конце укажи, кого необходимо уведомить"""

CHAT_SYSTEM = """Ты — «Умный ДСП-Регистратор», AI-ассистент дежурного по станции (ДСП) на железной дороге.

Помогаешь ДСП:
✅ Вести журнал движения поездов
✅ Классифицировать события (штатное / сбой / авария)  
✅ Находить инструкции по нормативным документам РЖД
✅ Анализировать расписание и задержки

Отвечай по-русски, профессионально и чётко. При критических ситуациях — немедленно давай инструкции.

История диалога:
{history}"""


# ── Менеджер памяти ───────────────────────────────────────────────────────────

class ConversationManager:
    """Управляет контекстным окном (Требование 6)."""

    def __init__(self, max_messages: int = 10):
        self.memory = ConversationBufferWindowMemory(
            k=max_messages,
            return_messages=True,
            memory_key="history",
        )

    def add_exchange(self, human: str, ai: str):
        self.memory.save_context({"input": human}, {"output": ai})

    def get_history_str(self) -> str:
        msgs = self.memory.load_memory_variables({}).get("history", [])
        if not msgs:
            return "Начало диалога."
        lines = []
        for msg in msgs:
            role = "ДСП" if isinstance(msg, HumanMessage) else "Ассистент"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def clear(self):
        self.memory.clear()


# ── Цепочка 1: Парсер сущностей ───────────────────────────────────────────────

class EntityParserChain:
    """
    Требование 7, Цепочка 1.
    Принимает сырой текст → возвращает структурированный JSON.
    """

    def __init__(self, llm):
        self.llm = llm
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(ENTITY_PARSER_SYSTEM),
            HumanMessagePromptTemplate.from_template("{raw_text}"),
        ])
        self.chain = prompt | llm | StrOutputParser()
    

    def parse(self, raw_text: str) -> dict:
        result_str = self.chain.invoke({"raw_text": raw_text})
        json_match = re.search(r'\{.*\}', result_str, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
            except json.JSONDecodeError:
                parsed = {}
        else:
            parsed = {}

        if not parsed.get("train_id"):
            m = re.search(r'поезд[а-я]*\s+(\d+)', raw_text.lower())
            if m:
                parsed["train_id"] = m.group(1)

        if not parsed.get("track"):
            m = re.search(r'пут[ьи]\s+(\d+)', raw_text.lower())
            if m:
                parsed["track"] = m.group(1)

        if not parsed.get("status"):
            parsed["status"] = raw_text[:50]

        return parsed if parsed else {
            "train_id": None, 
            "track": None,
            "status": raw_text[:50], 
            "issue": None,
            "action_required": False,
        }


# ── Цепочка 2: Генератор инструкций ──────────────────────────────────────────

class InstructionGeneratorChain:
    """
    Требование 7, Цепочка 2.
    Принимает описание ситуации + RAG-контекст → инструкции для ДСП.
    """

    def __init__(self, llm):
        self.llm = llm
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(INSTRUCTION_GENERATOR_SYSTEM),
            HumanMessagePromptTemplate.from_template(
                "Ситуация: {situation}\n\n"
                "Класс события: {event_class}\n\n"
                "Нормативный контекст:\n{rag_context}\n\n"
                "Сформулируй инструкции для ДСП."
            ),
        ])
        self.chain = prompt | llm | StrOutputParser()

    def generate(self, situation: str, event_class: str, rag_context: str) -> str:
        return self.chain.invoke({
            "situation": situation,
            "event_class": event_class,
            "rag_context": rag_context if rag_context else "Нормативный контекст не найден.",
        })


# ── Цепочка чата (с историей) ─────────────────────────────────────────────────

class ChatChain:
    """Общий чат с историей диалога (Требование 6)."""

    def __init__(self, llm, conversation_manager: ConversationManager):
        self.llm = llm
        self.conv = conversation_manager
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(CHAT_SYSTEM),
            HumanMessagePromptTemplate.from_template("{user_input}"),
        ])
        self.chain = prompt | llm | StrOutputParser()

    def chat(self, user_input: str) -> str:
        history = self.conv.get_history_str()
        response = self.chain.invoke({
            "history": history,
            "user_input": user_input,
        })
        self.conv.add_exchange(user_input, response)
        return response
