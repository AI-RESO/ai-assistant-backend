import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv
import re
from datetime import datetime, timezone, timedelta

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
)

# ============================================================
# 3. МОДЕЛИ ДАННЫХ
# ============================================================

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []
    sessionId: Optional[str] = None
    pageUrl: Optional[str] = None
    region: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    cta: Optional[dict] = None
    stage: str = "conversation"
    tokens_used: Dict[str, int] = {}

# ============================================================
# 4. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
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

def find_relevant_sections(query: str, knowledge_base: str, max_sections: int = 2):
    if not knowledge_base:
        return ""
    
    keywords = re.findall(r'[А-Яа-яA-Za-z0-9]{3,}', query.lower())
    
    if not keywords:
        return knowledge_base[:3000]
    
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
    has_phone_moscow = "+7 (499) 704-01-16" in reply or "704-01-16" in reply
    has_phone_regions = "+7 (499) 704-01-50" in reply or "704-01-50" in reply
    has_form = "forma-ai" in reply or "оставьте заявку" in reply.lower()
    
    cta_keywords = ["позвонить", "заявку", "оформление", "полис", "оформить"]
    has_cta = any(keyword in reply.lower() for keyword in cta_keywords)
    
    if not has_cta:
        return None
    
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
# 5. ЭНДПОИНТ ЧАТА
# ============================================================

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        time_info = get_time_info()
        region = request.region or "unknown"
        
        relevant_knowledge = find_relevant_sections(
            query=request.message,
            knowledge_base=KNOWLEDGE_BASE,
            max_sections=2
        )
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ:\nВремя: {time_info}\nРегион клиента: {region}"},
            {"role": "system", "content": "База знаний (только нужные разделы):\n\n" + relevant_knowledge}
        ]
        
        for msg in request.history[-10:]:
            messages.append(msg)
        
        messages.append({"role": "user", "content": request.message})
        
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages,
            temperature=0.3,
            max_tokens=1500,
            extra_body={"reasoning_effort": "low"}
        )
        
        reply = response.choices[0].message.content
        cta = extract_cta_from_reply(reply)
        
        if "здравствуйте" in reply.lower() or "привет" in reply.lower():
            stage = "greeting"
        elif "как вам удобнее" in reply.lower() or "какой вариант" in reply.lower():
            stage = "closing"
        else:
            stage = "conversation"
        
        print(f"📊 Токены: prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}, total={response.usage.total_tokens}")
        print(f"📍 Регион: {region}, Время: {'Рабочее' if is_working_hours() else 'Нерабочее'}")
        
        return ChatResponse(
            reply=reply,
            cta=cta,
            stage=stage,
            tokens_used={
                "prompt": response.usage.prompt_tokens,
                "completion": response.usage.completion_tokens,
                "total": response.usage.total_tokens,
            }
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
            stage="error"
        )

# ============================================================
# 6. ПРОВЕРКА РАБОТЫ
# ============================================================

@app.get("/")
async def root():
    return {"status": "ok", "message": "AI Assistant Backend is running!", "model": "deepseek-v4-flash"}

@app.get("/health")
async def health():
    is_working = is_working_hours()
    return {
        "status": "healthy",
        "working_hours": is_working,
        "time": datetime.now().astimezone(timezone(timedelta(hours=3))).strftime("%Y-%m-%d %H:%M:%S MSK")
    }

# ============================================================
# 7. ЗАПУСК
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)