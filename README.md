# BookVerse

[Посмотреть демонстрацию работы доп. задания](https://disk.yandex.ru/d/RwFJBBFvafxUYw) 

---

### Содержание
1. [Описание проекта](#описание-проекта)
2. [Выбор модели и принцип работы](#выбор-модели-и-принцип-работы)
   - [Почему выбрана модель Qwen2.5-3B-Instruct?](#почему-выбрана-модель-qwen25-3b-instruct)
   - [Принцип работы модели](#принцип-работы-модели)
3. [Краткая инструкция по запуску программы](#краткая-инструкция-по-запуску-программы)
   - [Требования](#требования)
   - [Шаги запуска](#шаги-запуска)

---

## Описание проекта
Проект представляет собой Telegram-бота, позволяющего пользователям сжимать предоставленный ими текст. Бот использует модель глубокого обучения для генерации краткого резюме текста. Доступны два уровня сжатия:

- **Сильное сжатие**: резюме ограничено 1-2 предложениями.
- **Слабое сжатие**: резюме представляет собой краткий абзац.

## Выбор модели и принцип работы

### Почему выбрана модель Qwen2.5-3B-Instruct?
- **Инструкционная специализация**: Модель Qwen2.5-3B-Instruct обучена на выполнении инструкций, что повышает её способность точно следовать пользовательским запросам и генерировать релевантные ответы.
- **Качество генерации**: Благодаря архитектуре и размеру (3 миллиарда параметров), модель способна создавать связные и осмысленные тексты, сохраняя ключевые детали исходного материала.
- **Доступность и интеграция**: Модель доступна через библиотеку Hugging Face Transformers, что упрощает её интеграцию в Python-проекты и обеспечивает широкую совместимость.
- **Поддержка длинного контекста**: Модель способна обрабатывать контексты длиной до 32 768 токенов, что позволяет работать с более объёмными текстами по сравнению с другими моделями аналогичного размера.

### Принцип работы модели
- **RoPE (Rotary Position Embedding)**: Метод позиционного кодирования, позволяющий модели учитывать порядок слов в предложении, что улучшает понимание последовательности текста.
- **SwiGLU**: Активационная функция, повышающая нелинейность модели и способствующая более точному захвату сложных зависимостей в данных.
- **RMSNorm**: Метод нормализации, стабилизирующий обучение и способствующий более эффективной генерации текста.
- **Attention QKV bias**: Механизм внимания с дополнительными смещениями, улучшающий способность модели фокусироваться на релевантных частях входного текста.

Модель принимает на вход текст и соответствующую инструкцию, объединяет их в специальный формат и генерирует ответ, следуя заданным параметрам и учитывая контекст. Это позволяет модели эффективно выполнять разнообразные задачи, требующие понимания и обработки естественного языка.

## Краткая инструкция по запуску программы

### Требования:
- Python версии 3.8 или выше.
- Macbook Pro 2021 или новее.
- M1 Pro или лучше.
- 16GB объединённой ОЗУ или больше.
- Токен Telegram-бота, полученный от BotFather.

### Шаги запуска:
1. **Клонируйте репозиторий** или скопируйте код программы на ваш локальный компьютер.
2. **Установите зависимости** из файла `requirements.txt`:
   - В терминале выполните команду: `pip install -r requirements.txt`
3. **Настройте токен бота**:
   - В файле кода найдите строку: `API_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"`
   - Замените `"YOUR_TELEGRAM_BOT_TOKEN"` на ваш реальный токен, полученный от BotFather.
4. **Запустите скрипт**:
   - В терминале выполните команду: `python bot.py`
5. **Начните взаимодействие с ботом**:
   - Откройте Telegram и найдите вашего бота по имени.
   - Отправьте команду `/start`.
   - Следуйте инструкциям бота для отправки текста и выбора уровня сжатия (сильное или слабое сжатие).

Теперь вы готовы использовать бота для сжатия текста, следуя инструкциям в Telegram.
