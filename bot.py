import asyncio
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

# Инициализация бота
API_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
bot = Bot(token=API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

# Загрузка модели и токенизатора
model_name = "Qwen/Qwen2.5-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Состояния для FSM
class SummarizationStates(StatesGroup):
    waiting_for_text = State()
    waiting_for_level = State()

# Функция генерации резюме
def generate_summary(text, level="weak"):
    """
    Генерирует резюме текста на основе указанного уровня сжатия.

    :param text: Текст для сжатия
    :param level: Уровень сжатия (strong или weak)
    :return: Сжатый текст
    """
    
    system_prompt = "Ты бот который максимально кратко и понятно излагает текст. Важно передать основную мысль и сохраять ключевые детали и события."
    user_prompt = (
        f"Прочитай текст и передай основную мысль. Ответ должен быть обязательно до 1-2 предложений.\n\nТекст:\n{text}"
        if level == "strong"
        else f"Прочитай текст и обязательно сократи его до 1 краткого абзаца, сохраняя ключевые детали и события.\n\nТекст:\n{text}"
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    
    model_input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([model_input_text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# Клавиатуры
compression_keyboard = InlineKeyboardMarkup(
    inline_keyboard=[
        [InlineKeyboardButton(text="Сильное сжатие (1-2 предложения)", callback_data="strong")],
        [InlineKeyboardButton(text="Слабое сжатие (краткий абзац)", callback_data="weak")]
    ]
)

new_text_keyboard = InlineKeyboardMarkup(
    inline_keyboard=[
        [InlineKeyboardButton(text="Загрузить новый текст", callback_data="new_text")]
    ]
)

# Обработчик команды /start
@dp.message(Command("start"))
async def start_command(message: types.Message, state: FSMContext):
    await message.answer("Привет! Отправьте мне текст, который нужно сжать.")
    await state.set_state(SummarizationStates.waiting_for_text)

# Обработчик для получения текста
@dp.message(SummarizationStates.waiting_for_text)
async def get_text(message: types.Message, state: FSMContext):
    await state.update_data(text=message.text)
    await message.answer("Выберите уровень сжатия:", reply_markup=compression_keyboard)
    await state.set_state(SummarizationStates.waiting_for_level)

# Обработчик для выбора уровня сжатия и генерации резюме
@dp.callback_query(SummarizationStates.waiting_for_level)
async def summarize_text(callback_query: types.CallbackQuery, state: FSMContext):
    await callback_query.answer()  
    
    data = await state.get_data()
    text = data.get("text")

    if not text:
        await callback_query.message.answer("Пожалуйста, отправьте текст для сжатия.")
        await state.set_state(SummarizationStates.waiting_for_text)
        return

    # Определяем уровень сжатия
    level = "strong" if callback_query.data == "strong" else "weak"
    # Обновляем сообщение, чтобы показать, что идет генерация
    await callback_query.message.edit_text("Генерирую резюме...")

    await bot.send_chat_action(callback_query.from_user.id, action="typing")

    # Генерация резюме
    summary = generate_summary(text, level)

    # Отправляем результат с новой клавиатурой
    await callback_query.message.edit_text(f"📜 Резюме:\n\n{summary}", reply_markup=new_text_keyboard)
    
    # Очищаем состояние для нового ввода
    await state.set_state(SummarizationStates.waiting_for_text)

# Обработчик для загрузки нового текста
@dp.callback_query(F.data == "new_text")
async def new_text_handler(callback_query: types.CallbackQuery, state: FSMContext):
    await callback_query.answer() 
    await callback_query.message.answer("Отправьте новый текст для сжатия.")
    await state.set_state(SummarizationStates.waiting_for_text)

# Запуск бота
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
