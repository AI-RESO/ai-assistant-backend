import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv
import re
from datetime import datetime, timezone, timedelta
import json
from functools import lru_cache

load_dotenv()

# ============================================================
# 0. ПРОВЕРКА КЛЮЧА API
# ============================================================

api_key = os.environ.get("DEEPSEEK_API_KEY")
if not api_key:
    print("⚠️ ОШИБКА: DEEPSEEK_API_KEY не найден в .env!")
    print("📝 Создайте файл .env с содержимым: DEEPSEEK_API_KEY=sk-ваш_ключ")
    exit(1)

# ============================================================
# 1. ЗАГРУЗКА ПРОМПТА И БАЗЫ ЗНАНИЙ ИЗ ФАЙЛОВ
# ============================================================

def load_system_prompt():
    try:
        with open('system_prompt.txt', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print("⚠️ Файл system_prompt.txt не найден!")
        return "Ты — AI-консультант РЕСО-Гарантия."

def load_knowledge_base():
    try:
        with open('knowledge_base.txt', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print("⚠️ Файл knowledge_base.txt не найден!")
        return ""

SYSTEM_PROMPT = load_system_prompt()
KNOWLEDGE_BASE = load_knowledge_base()

print(f"✅ Системный промпт загружен: {len(SYSTEM_PROMPT)} символов")
print(f"✅ База знаний загружена: {len(KNOWLEDGE_BASE)} символов")

# ============================================================
# 2. НАСТРОЙКИ
# ============================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com/v1",
    timeout=30.0,
    max_retries=2,
)

# ============================================================
# 3. МОДЕЛИ ДАННЫХ
# ============================================================

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []
    sessionId: Optional[str] = None
    pageUrl: Optional[str] = None
    region: Optional[str] = None          # Передаётся из виджета
    timezone: Optional[str] = None        # Передаётся из виджета (часовой пояс)
    stream: bool = False

class ChatResponse(BaseModel):
    reply: str
    cta: Optional[dict] = None
    stage: str = "conversation"
    tokens_used: Dict[str, int] = {}
    detected_region: Optional[str] = None  # Для отладки

# ============================================================
# 4. КЕШИРОВАНИЕ
# ============================================================

# Кеш ответов на популярные вопросы
answer_cache = {}
CACHE_TTL = 3600  # 1 час

def get_cached_answer(message: str, region: str = "unknown") -> Optional[str]:
    cache_key = f"{message.lower().strip()}:{region}"
    if cache_key in answer_cache:
        cache_time, reply = answer_cache[cache_key]
        if (datetime.now() - cache_time).seconds < CACHE_TTL:
            return reply
        else:
            del answer_cache[cache_key]
    return None

def set_cached_answer(message: str, reply: str, region: str = "unknown"):
    cache_key = f"{message.lower().strip()}:{region}"
    answer_cache[cache_key] = (datetime.now(), reply)

# ============================================================
# 5. ОПРЕДЕЛЕНИЕ РЕГИОНА ПО ЧАСОВОМУ ПОЯСУ
# ============================================================

# Часовые пояса, которые соответствуют Москве и Московской области
MOSCOW_TIMEZONES = {
    "Europe/Moscow",
    "Europe/Volgograd",
    "Europe/Saratov",
    "Europe/Ulyanovsk",
    "Europe/Samara",
    "Europe/Kirov",
}

def detect_region_by_timezone(timezone: str) -> str:
    """
    Определяет регион по часовому поясу браузера.
    Возвращает: "moscow" или "region"
    """
    if not timezone:
        return "unknown"
    
    # Очищаем от лишних пробелов
    tz = timezone.strip()
    
    if tz in MOSCOW_TIMEZONES:
        return "moscow"
    
    # Дополнительная проверка: если часовой пояс начинается с Europe/
    # и не входит в список московских — скорее всего это регион
    if tz.startswith("Europe/"):
        return "region"
    
    # Все остальные часовые пояса (Азия, другие) — регион
    return "region"

def detect_region_from_request(request: Request, chat_request: ChatRequest) -> str:
    """
    Комбинированное определение региона:
    1. Если передан region в запросе — используем его
    2. Если передан timezone в запросе — определяем по нему
    3. Иначе — "unknown"
    """
    # 1. Используем переданный регион (если есть)
    if chat_request.region and chat_request.region != "unknown":
        return chat_request.region
    
    # 2. Определяем по часовому поясу
    if chat_request.timezone:
        detected = detect_region_by_timezone(chat_request.timezone)
        if detected != "unknown":
            return detected
    
    # 3. Если ничего не помогло — возвращаем "unknown"
    return "unknown"

# ============================================================
# 6. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def is_working_hours() -> bool:
    now = datetime.now().astimezone(timezone(timedelta(hours=3)))
    hour = now.hour
    day = now.weekday()
    
    if day == 6:
        return False
    if day == 5:
        return 10 <= hour < 16
    return 10 <= hour < 20

def get_time_info() -> str:
    is_working = is_working_hours()
    status = "Рабочее" if is_working else "Нерабочее"
    return f"ТЕКУЩЕЕ ВРЕМЯ: {status}. {'Можно предлагать звонок.' if is_working else 'НЕЛЬЗЯ предлагать звонок, только заявку.'}"

# Кешированный поиск по базе знаний
@lru_cache(maxsize=256)
def find_relevant_sections_cached(query: str, max_sections: int = 2):
    return find_relevant_sections(query, KNOWLEDGE_BASE, max_sections)

def find_relevant_sections(query: str, knowledge_base: str, max_sections: int = 2):
    if not knowledge_base:
        return ""
    
    keywords = re.findall(r'[А-Яа-яA-Za-z0-9]{3,}', query.lower())
    
    if not keywords:
        return knowledge_base[:5000]
    
    sections = re.split(r'(?=^={3,} )', knowledge_base, flags=re.MULTILINE)
    
    scored_sections = []
    for section in sections:
        if len(section.strip()) < 100:
            continue
        
        section_lower = section.lower()
        score = 0
        
        for keyword in keywords:
            count = section_lower.count(keyword)
            if count > 0:
                weight = count * 2
                if keyword in section_lower[:200]:
                    weight += 5
                if keyword in section_lower[:500]:
                    weight += 3
                score += weight
        
        if score > 0:
            scored_sections.append((score, section))
    
    scored_sections.sort(key=lambda x: x[0], reverse=True)
    top_sections = [s[1] for s in scored_sections[:max_sections]]
    
    if not top_sections:
        for section in sections:
            if "О КОМПАНИИ" in section.upper() or "КОНТАКТЫ" in section.upper():
                top_sections.append(section)
                break
        if not top_sections:
            top_sections = [knowledge_base[:5000]]
    
    return "\n\n".join(top_sections)

def extract_cta_from_reply(reply: str) -> Optional[dict]:
    if not any(keyword in reply.lower() for keyword in ["позвонить", "заявку", "оформление", "полис", "оформить"]):
        return None
    
    has_phone_moscow = "+7 (499) 704-01-16" in reply or "704-01-16" in reply
    has_phone_regions = "+7 (499) 704-01-50" in reply or "704-01-50" in reply
    has_form = "forma-ai" in reply or "оставьте заявку" in reply.lower()
    
    actions = []
    
    if has_phone_moscow:
        actions.append({
            "type": "phone",
            "label": "📞 Позвонить (Москва)",
            "value": "+74997040116",
            "description": "для Москвы"
        })
    
    if has_phone_regions:
        actions.append({
            "type": "phone",
            "label": "📞 Позвонить (Регионы)",
            "value": "+74997040150",
            "description": "для регионов"
        })
    
    if has_form:
        actions.append({
            "type": "form",
            "label": "📝 Оставить заявку",
            "value": "https://resostrahovka.ru/forma-ai/",
            "description": "мы перезвоним сами"
        })
    
    if not actions:
        if "позвонить" in reply.lower():
            actions.append({
                "type": "phone",
                "label": "📞 Позвонить агенту",
                "value": "+74997040116",
                "description": "для Москвы"
            })
            actions.append({
                "type": "phone",
                "label": "📞 Позвонить агенту",
                "value": "+74997040150",
                "description": "для регионов"
            })
        
        if "заявку" in reply.lower():
            actions.append({
                "type": "form",
                "label": "📝 Оставить заявку",
                "value": "https://resostrahovka.ru/forma-ai/",
                "description": "мы перезвоним сами"
            })
    
    if not actions:
        return None
    
    return {
        "title": "Оформить полис можно за 10 минут",
        "subtitle": "Выберите удобный способ",
        "actions": actions
    }

# ============================================================
# 7. ОСНОВНОЙ ЭНДПОИНТ ЧАТА
# ============================================================

@app.post("/api/chat")
async def chat(request: Request, chat_request: ChatRequest):
    try:
        # 1. Определяем регион
        detected_region = detect_region_from_request(request, chat_request)
        
        # 2. Проверяем кеш (с учётом региона)
        cached_reply = get_cached_answer(chat_request.message, detected_region)
        if cached_reply:
            print(f"⚡ Кеш-хит для: {chat_request.message[:30]}... (регион: {detected_region})")
            cta = extract_cta_from_reply(cached_reply)
            return ChatResponse(
                reply=cached_reply,
                cta=cta,
                stage="conversation",
                tokens_used={"cached": True},
                detected_region=detected_region
            )
        
        # 3. Получаем информацию о времени
        time_info = get_time_info()
        region = detected_region if detected_region != "unknown" else "region"
        
        # 4. Поиск по базе знаний (с кешированием)
        relevant_knowledge = find_relevant_sections_cached(
            query=chat_request.message,
            max_sections=2
        )
        
        # 5. Полная история — 10 сообщений
        history = chat_request.history[-10:] if chat_request.history else []
        
        # 6. Формируем сообщения
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ:\nВремя: {time_info}\nРегион клиента: {region}"},
            {"role": "system", "content": "База знаний (только нужные разделы):\n\n" + relevant_knowledge}
        ]
        
        for msg in history:
            messages.append(msg)
        
        messages.append({"role": "user", "content": chat_request.message})
        
        # 7. Вызов API
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages,
            temperature=0.3,
            max_tokens=1500,
            extra_body={"reasoning_effort": "low"}
        )
        
        reply = response.choices[0].message.content
        
        # 8. Сохраняем в кеш (с учётом региона)
        set_cached_answer(chat_request.message, reply, detected_region)
        
        # 9. Извлекаем CTA
        cta = extract_cta_from_reply(reply)
        
        # 10. Определяем стадию
        if "здравствуйте" in reply.lower() or "привет" in reply.lower():
            stage = "greeting"
        elif "как вам удобнее" in reply.lower() or "какой вариант" in reply.lower():
            stage = "closing"
        else:
            stage = "conversation"
        
        print(f"📊 Токены: prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}")
        print(f"📍 Регион: {detected_region} (исходный: {chat_request.timezone})")
        
        return ChatResponse(
            reply=reply,
            cta=cta,
            stage=stage,
            tokens_used={
                "prompt": response.usage.prompt_tokens,
                "completion": response.usage.completion_tokens,
                "total": response.usage.total_tokens,
            },
            detected_region=detected_region
        )
    
    except Exception as e:
        print(f"Ошибка: {e}")
        return ChatResponse(
            reply="Извините, произошла ошибка. Пожалуйста, позвоните нашему агенту:\n\n📞 **Москва:** [+7 (499) 704-01-16](tel:+74997040116)\n📞 **Регионы:** [+7 (499) 704-01-50](tel:+74997040150)\n\nИли [📝 оставьте заявку](https://resostrahovka.ru/forma-ai/) — мы перезвоним вам сами.",
            cta={
                "title": "Свяжитесь с нами",
                "subtitle": "Выберите удобный способ",
                "actions": [
                    {"type": "phone", "label": "📞 Позвонить (Москва)", "value": "+74997040116", "description": "для Москвы"},
                    {"type": "phone", "label": "📞 Позвонить (Регионы)", "value": "+74997040150", "description": "для регионов"},
                    {"type": "form", "label": "📝 Оставить заявку", "value": "https://resostrahovka.ru/forma-ai/", "description": "мы перезвоним сами"}
                ]
            },
            stage="error",
            detected_region="unknown"
        )

# ============================================================
# 8. ЭНДПОИНТ ДЛЯ СТРИМИНГА
# ============================================================

@app.post("/api/chat/stream")
async def chat_stream(request: Request, chat_request: ChatRequest):
    """Стриминг с сохранением полного качества ответа"""
    
    async def generate():
        try:
            # Определяем регион
            detected_region = detect_region_from_request(request, chat_request)
            region = detected_region if detected_region != "unknown" else "region"
            
            # Проверяем кеш
            cached_reply = get_cached_answer(chat_request.message, detected_region)
            if cached_reply:
                for i in range(0, len(cached_reply), 20):
                    chunk = cached_reply[i:i+20]
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            # Получаем информацию
            time_info = get_time_info()
            
            # Поиск по БЗ (с кешированием)
            relevant_knowledge = find_relevant_sections_cached(
                query=chat_request.message,
                max_sections=2
            )
            
            # Полная история — 10 сообщений
            history = chat_request.history[-10:] if chat_request.history else []
            
            # Формируем сообщения
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "system", "content": f"ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ:\nВремя: {time_info}\nРегион клиента: {region}"},
                {"role": "system", "content": "База знаний (только нужные разделы):\n\n" + relevant_knowledge}
            ]
            
            for msg in history:
                messages.append(msg)
            
            messages.append({"role": "user", "content": chat_request.message})
            
            # Стриминг
            stream = client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=messages,
                temperature=0.3,
                max_tokens=1500,
                extra_body={"reasoning_effort": "low"},
                stream=True
            )
            
            full_reply = ""
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_reply += content
                    yield f"data: {json.dumps({'content': content})}\n\n"
            
            # Сохраняем в кеш
            set_cached_answer(chat_request.message, full_reply, detected_region)
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            print(f"Ошибка стриминга: {e}")
            error_msg = "Извините, произошла ошибка. Пожалуйста, попробуйте позже."
            yield f"data: {json.dumps({'content': error_msg})}\n\n"
            yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

# ============================================================
# 9. ПРОВЕРКА РАБОТЫ
# ============================================================

@app.get("/")
async def root():
    return {
        "status": "ok", 
        "message": "AI Assistant Backend is running!", 
        "model": "deepseek-v4-flash",
        "quality": "full",
        "cache": "enabled",
        "region_detection": "timezone"
    }

@app.get("/health")
async def health():
    is_working = is_working_hours()
    return {
        "status": "healthy",
        "working_hours": is_working,
        "time": datetime.now().astimezone(timezone(timedelta(hours=3))).strftime("%Y-%m-%d %H:%M:%S MSK"),
        "cache_size": len(answer_cache),
        "quality_mode": "full",
        "region_detection": "timezone"
    }

@app.delete("/cache")
async def clear_cache():
    """Очистка кеша (для администрирования)"""
    answer_cache.clear()
    find_relevant_sections_cached.cache_clear()
    return {"status": "ok", "message": "Cache cleared"}

# ============================================================
# 10. ЗАПУСК
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)