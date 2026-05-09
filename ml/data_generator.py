"""
=============================================================================
Генератор реалистичного мультимодального датасета для обучения NER-модели
на именах медиафайлов (фильмы, сериалы, аудиозаписи).

Тема ВКР: «Индексация медиаконтента и обогащение метаданных
           с использованием интеллектуального анализа данных»

Автор:  Феденко Никита Александрович
Группа: ИД 23.1/Б3-22
Год:    2026

Описание:
    Версия 3.0. Расширение базового словаря и шаблонов:
        - ~100 фильмов (английских + русских + транслит)
        - ~80 музыкантов (международных + русских)
        - 75 уникальных годов (1950-2024)
        - 25+ форматов качества (1080p/4K/HEVC/FLAC/...)
        - ~15 различных шаблонов имён файлов
        - сериалы (с обозначениями SxxEyy)
        - edge-cases (короткие имена, отсутствие отдельных сущностей)

    Базовый словарь сформирован из публичных источников:
    IMDb Top-250, Last.fm Top, открытые каталоги российского
    кинопроизводства и эстрады. Шаблоны имён файлов основаны
    на реальных практиках именования в P2P-сообществе
    (RARBG, YIFY, российские раздачи) и стандартах файловых
    систем (DSC_, IMG_, и т.п.).

    Формат выходного CSV: text,tags
        text — строка с пробелами как разделителями слов
               (твой ner_predictor.py уже преобразует ./_/-
               в пробелы в препроцессинге)
        tags — список IOB-меток через запятую, выровненный
               по словам в text (после text.split())

    Сохраняемая разметка (IOB scheme):
        B-TITLE / I-TITLE   — название произведения
        B-YEAR              — год выпуска
        B-QUALITY / I-QUALITY — техническое качество
        B-ARTIST / I-ARTIST — исполнитель
        O                   — служебные/нерелевантные токены
=============================================================================
"""

import csv
import random
from pathlib import Path

# =========================================================================
# КОНФИГУРАЦИЯ
# =========================================================================

RANDOM_SEED = 42

# Распределение целевого датасета по типам
SAMPLES_DISTRIBUTION = {
    "movie_en":      9000,   # англоязычные фильмы (основа)
    "movie_ru_lat":  2500,   # русские фильмы в транслите
    "movie_ru_cyr":  1500,   # русские фильмы кириллицей (для мультиязычной v2)
    "tv_series":     2000,   # сериалы
    "audio_en":      2500,   # англоязычная музыка
    "audio_ru_lat":  1000,   # русская музыка в транслите
    "audio_ru_cyr":   500,   # русская музыка кириллицей
    "edge_case":     1000,   # короткие/минималистичные имена
}

# =========================================================================
# СЛОВАРИ
# =========================================================================

# --- Англоязычные фильмы ---
MOVIES_EN = [
    "The Matrix", "Inception", "The Dark Knight", "Interstellar", "Pulp Fiction",
    "Avatar", "Fight Club", "Forrest Gump", "Gladiator", "Titanic",
    "The Godfather", "Goodfellas", "Casino", "Heat", "Reservoir Dogs",
    "Kill Bill", "Django Unchained", "Inglourious Basterds", "The Departed",
    "Saving Private Ryan", "Schindlers List", "Apocalypse Now", "Full Metal Jacket",
    "Dunkirk", "Tenet", "Memento", "The Prestige", "Batman Begins",
    "The Dark Knight Rises", "The Avengers", "Iron Man", "Captain America",
    "Thor", "Doctor Strange", "Black Panther", "Guardians of the Galaxy",
    "Deadpool", "Logan", "Joker", "Wonder Woman", "Aquaman", "Man of Steel",
    "Spider Man", "Avengers Endgame", "Avengers Infinity War",
    "The Lord of the Rings", "The Hobbit", "The Two Towers", "Return of the King",
    "Star Wars", "The Empire Strikes Back", "Return of the Jedi",
    "Indiana Jones", "Back to the Future", "Jurassic Park", "Jurassic World",
    "The Terminator", "Terminator Genisys", "Alien", "Aliens", "Predator",
    "The Lion King", "Frozen", "Toy Story", "Up", "WALL E", "Inside Out",
    "Coco", "Soul", "Encanto", "Moana", "Finding Nemo",
    "The Shawshank Redemption", "Dune", "Oppenheimer", "Barbie", "Parasite",
    "Whiplash", "La La Land", "The Revenant", "Birdman", "Spotlight",
    "Mad Max Fury Road", "Blade Runner", "Blade Runner 2049",
    "Ex Machina", "Arrival", "Annihilation", "Gravity", "The Martian",
    "1917", "Once Upon a Time in Hollywood", "No Time to Die", "Skyfall",
    "Casino Royale", "John Wick", "John Wick Chapter 2", "John Wick Chapter 3",
    "The Hateful Eight", "Hacksaw Ridge", "American Sniper", "Lone Survivor",
]

# --- Российские фильмы (кириллица) ---
MOVIES_RU_CYR = [
    "Брат", "Брат 2", "Война", "Левиафан", "Сталинград", "Викинг",
    "Движение вверх", "Легенда 17", "Холоп", "Ёлки",
    "Иван Васильевич меняет профессию", "Бриллиантовая рука",
    "Кавказская пленница", "Операция Ы", "Москва слезам не верит",
    "Ирония судьбы", "Девчата", "Любовь и голуби", "Двенадцать",
    "Утомлённые солнцем", "Сибирский цирюльник", "Турецкий гамбит",
    "Адмиралъ", "Стиляги", "Высоцкий", "Метро", "Майор", "Дурак",
    "Дылда", "Нелюбовь", "Юрьев день", "Аритмия", "Звезда", "Битва",
    "Бой с тенью", "Жмурки", "Бумер", "Антикиллер", "Дневной дозор",
    "Ночной дозор", "Чёрная молния", "Чебурашка",
]

# --- Российские фильмы (транслит) ---
MOVIES_RU_LAT = [
    "Brat", "Brat 2", "Voyna", "Leviafan", "Stalingrad", "Vikingo",
    "Dvizhenie Vverh", "Legenda 17", "Holop", "Yolki",
    "Ivan Vasilevich Menyaet Professiyu", "Brilliantovaya Ruka",
    "Kavkazskaya Plennitsa", "Operatsiya Y", "Moskva Slezam Ne Verit",
    "Ironiya Sudby", "Devchata", "Lyubov i Golubi", "Dvenadtsat",
    "Utomlennye Solntsem", "Sibirskiy Tsiryulnik", "Turetskiy Gambit",
    "Admiral", "Stilyagi", "Vysotskiy", "Metro", "Mayor", "Durak",
    "Dylda", "Nelyubov", "Yurev Den", "Aritmiya", "Zvezda", "Bitva",
    "Boy s Tenyu", "Zhmurki", "Boomer", "Antikiller", "Dnevnoy Dozor",
    "Nochnoy Dozor", "Chernaya Molniya", "Cheburashka",
]

# --- Англоязычные сериалы ---
TV_SERIES_EN = [
    "Breaking Bad", "Better Call Saul", "Game of Thrones", "House of the Dragon",
    "The Sopranos", "The Wire", "True Detective", "Westworld",
    "Stranger Things", "The Mandalorian", "The Boys", "The Witcher",
    "Sherlock", "Peaky Blinders", "Vikings", "The Last of Us",
    "Black Mirror", "Mr Robot", "Fargo", "Yellowstone",
    "Succession", "Euphoria", "House MD", "Friends", "How I Met Your Mother",
    "The Office", "Brooklyn Nine Nine", "Seinfeld", "Lost", "Dexter",
    "Prison Break", "24", "Homeland", "Suits", "Mindhunter",
    "Narcos", "Money Heist", "Squid Game", "Dark", "The Crown",
]

# --- Российские сериалы (для разнообразия) ---
TV_SERIES_RU_LAT = [
    "Brigada", "Sled", "Slovo Patsana", "Olga", "Univer",
    "Interny", "Kuhnya", "Fiziruk", "Sklifosovskiy", "Mayor Sokolov",
    "Likvidatsiya", "Esenin", "Doctor Richter", "Mazhor",
]

# --- Иностранные музыканты ---
ARTISTS_EN = [
    "Hans Zimmer", "Queen", "The Beatles", "Daft Punk", "Nirvana",
    "Pink Floyd", "David Bowie", "Eminem", "Coldplay", "Radiohead",
    "Linkin Park", "Metallica", "AC DC", "Led Zeppelin",
    "The Rolling Stones", "Michael Jackson", "Madonna", "Beyonce",
    "Adele", "Ed Sheeran", "Taylor Swift", "Drake", "Kanye West",
    "Jay Z", "Kendrick Lamar", "Tupac", "Snoop Dogg", "Dr Dre",
    "Lady Gaga", "Rihanna", "Bruno Mars", "The Weeknd", "Post Malone",
    "Billie Eilish", "Dua Lipa", "Imagine Dragons", "Maroon 5",
    "Foo Fighters", "Red Hot Chili Peppers", "System of a Down",
    "Rammstein", "Iron Maiden", "Black Sabbath", "Deep Purple",
    "Bob Dylan", "Johnny Cash", "Frank Sinatra", "Elvis Presley",
    "Beatles", "Pink", "Sting", "U2", "Oasis",
]

# --- Российские музыканты (кириллица) ---
ARTISTS_RU_CYR = [
    "Кино", "ДДТ", "Машина Времени", "Аквариум", "Ария", "Алиса",
    "Сектор Газа", "Гражданская Оборона", "Мумий Тролль", "Сплин",
    "Земфира", "Звери", "Ленинград", "Король и Шут", "Би 2",
    "Ночные Снайперы", "Многоточие", "Каста", "Каспийский Груз",
    "Баста", "Тимати", "Валерий Меладзе", "Григорий Лепс",
    "Стас Михайлов", "Михаил Круг", "Скриптонит", "Macan", "Markul",
    "Big Baby Tape", "Хабиб", "Полина Гагарина", "ВИА Гра",
]

# --- Российские музыканты (транслит) ---
ARTISTS_RU_LAT = [
    "Kino", "DDT", "Mashina Vremeni", "Akvarium", "Ariya", "Alisa",
    "Sektor Gaza", "Grazhdanskaya Oborona", "Mumiy Troll", "Splin",
    "Zemfira", "Zveri", "Leningrad", "Korol i Shut", "Bi 2",
    "Nochnye Snaypery", "Mnogotochie", "Kasta", "Kaspiyskiy Gruz",
    "Basta", "Timati", "Valeriy Meladze", "Grigoriy Leps",
    "Stas Mihaylov", "Mihail Krug", "Skriptonit",
]

# --- Названия треков (англ) ---
TRACKS_EN = [
    "Bohemian Rhapsody", "Time", "Smells Like Teen Spirit", "Let It Be",
    "Lose Yourself", "Heroes", "Starboy", "Money", "Imagine",
    "Stairway to Heaven", "Hey Jude", "Yesterday", "Like a Rolling Stone",
    "Hotel California", "Sweet Child O Mine", "Wonderwall", "Creep",
    "Boulevard of Broken Dreams", "Numb", "In the End", "Crawling",
    "One", "Master of Puppets", "Enter Sandman", "Nothing Else Matters",
    "Highway to Hell", "Thunderstruck", "Back in Black", "Smoke on the Water",
    "Comfortably Numb", "Another Brick in the Wall", "Wish You Were Here",
    "Dont Stop Believin", "Eye of the Tiger", "Africa", "Take On Me",
    "Billie Jean", "Thriller", "Beat It", "Smooth Criminal",
    "Rolling in the Deep", "Someone Like You", "Hello", "Shape of You",
    "Perfect", "Thinking Out Loud", "Bad Guy", "Blinding Lights",
    "Watermelon Sugar", "Levitating", "Save Your Tears",
]

# --- Названия треков (русские, кириллица) ---
TRACKS_RU_CYR = [
    "Группа Крови", "Звезда по Имени Солнце", "Перемен", "Восьмиклассница",
    "Мама Анархия", "Кукушка", "Хочу Перемен", "Видели Ночь",
    "Это Не Любовь", "Когда Твоя Девушка Больна", "Алюминиевые Огурцы",
    "Дерево", "Печаль", "Невидимка", "Малыш", "Нам с Тобой",
    "Песня без Слов", "Вечер", "Бошетунмай", "Ночь",
    "Москва", "Питер", "Менуэт", "В Питере Пить", "WWW",
    "Ягода Малинка", "Кайфуем", "Делай Громче",
]

# --- Названия треков (транслит) ---
TRACKS_RU_LAT = [
    "Gruppa Krovi", "Zvezda po Imeni Solntse", "Peremen", "Vosmiklassnitsa",
    "Mama Anarhia", "Kukushka", "Hochu Peremen", "Videli Noch",
    "Eto Ne Lyubov", "Kogda Tvoya Devushka Bolna", "Alyuminievye Ogurtsy",
    "Derevo", "Pechal", "Nevidimka", "Malysh", "Nam s Toboy",
    "Pesnya bez Slov", "Vecher", "Boshetunmay", "Noch",
    "Yagoda Malinka", "Kayfuem", "Delay Gromche",
]

# --- Годы (1950-2024) ---
YEARS = [str(y) for y in range(1950, 2025)]

# --- Качества для видео ---
QUALITIES_VIDEO_RES = ["480p", "720p", "1080p", "1440p", "2160p", "4K", "8K"]
QUALITIES_VIDEO_SRC = ["BluRay", "WEBRip", "WEB DL", "HDRip", "DVDRip", "BDRemux",
                       "REMUX", "HDTV", "BDRip"]
QUALITIES_VIDEO_CDC = ["x264", "x265", "H 264", "H 265", "HEVC", "AVC", "DivX", "XviD"]

# --- Качества для аудио ---
QUALITIES_AUDIO = ["FLAC", "Lossless", "320kbps", "256kbps", "192kbps",
                   "128kbps", "MP3", "OGG", "ALAC", "M4A", "DSD", "AAC"]

# --- Релиз-группы (для шаблонов с -GROUP) ---
RELEASE_GROUPS = ["RARBG", "YIFY", "EVO", "MZABI", "FGT", "PSA", "SPARKS",
                  "GECKOS", "CMRG", "KOGi", "TGx", "CHD", "WiKi"]


# =========================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================================================================

def make_iob(words, label):
    """
    Преобразует список слов в IOB-разметку для конкретной сущности.
    Например: ['The', 'Matrix'], 'TITLE' -> ['B-TITLE', 'I-TITLE']
    """
    if not words:
        return []
    return [f"B-{label}"] + [f"I-{label}"] * (len(words) - 1)


def random_video_quality():
    """
    Возвращает случайную комбинацию характеристик видеокачества как
    список одиночных токенов. Использует .split() на каждом значении,
    чтобы многословные форматы вроде "WEB DL", "H 264" корректно
    превращались в отдельные токены и не ломали выравнивание разметки.
    """
    parts = []
    if random.random() > 0.1:
        parts.extend(random.choice(QUALITIES_VIDEO_RES).split())
    if random.random() > 0.4:
        parts.extend(random.choice(QUALITIES_VIDEO_SRC).split())
    if random.random() > 0.6:
        parts.extend(random.choice(QUALITIES_VIDEO_CDC).split())
    if not parts:
        parts.extend(random.choice(QUALITIES_VIDEO_RES).split())
    return parts


# =========================================================================
# ШАБЛОНЫ ИМЁН ФАЙЛОВ
# =========================================================================

def template_movie_classic(title, year, quality_parts, group=None):
    """
    Классический формат: Title Year Quality [GROUP]
    Пример: 'The Dark Knight 2008 1080p BluRay x264 RARBG'
    """
    title_words = title.split()
    text = title_words + [year] + quality_parts
    tags = (
        make_iob(title_words, "TITLE")
        + make_iob([year], "YEAR")
        + make_iob(quality_parts, "QUALITY")
    )
    if group:
        text.append(group)
        tags.append("O")
    return text, tags


def template_movie_year_first(title, year, quality_parts):
    """
    Год впереди: Year Title Quality
    Пример: '2014 Interstellar 4K WEB DL'
    """
    title_words = title.split()
    text = [year] + title_words + quality_parts
    tags = (
        make_iob([year], "YEAR")
        + make_iob(title_words, "TITLE")
        + make_iob(quality_parts, "QUALITY")
    )
    return text, tags


def template_movie_minimal(title, year):
    """
    Только название и год, без качества.
    Пример: 'Joker 2019'
    """
    title_words = title.split()
    text = title_words + [year]
    tags = make_iob(title_words, "TITLE") + make_iob([year], "YEAR")
    return text, tags


def template_movie_no_year(title, quality_parts):
    """
    Только название и качество, без года.
    Пример: 'Inception 1080p BluRay'
    """
    title_words = title.split()
    text = title_words + quality_parts
    tags = make_iob(title_words, "TITLE") + make_iob(quality_parts, "QUALITY")
    return text, tags


def template_movie_only_title(title):
    """
    Только название (короткое имя файла).
    Пример: 'Avatar'
    """
    title_words = title.split()
    text = list(title_words)
    tags = make_iob(title_words, "TITLE")
    return text, tags


def template_tv_series(title, year, quality_parts):
    """
    Сериал с обозначением сезона/эпизода S01E01.
    Пример: 'Breaking Bad S01E05 1080p WEBRip'
    """
    title_words = title.split()
    season = random.randint(1, 8)
    episode = random.randint(1, 24)
    se_marker = f"S{season:02d}E{episode:02d}"

    text = title_words + [se_marker] + [year] + quality_parts
    tags = (
        make_iob(title_words, "TITLE")
        + ["O"]
        + make_iob([year], "YEAR")
        + make_iob(quality_parts, "QUALITY")
    )
    return text, tags


def template_tv_series_no_year(title, quality_parts):
    """Сериал без года: 'Stranger Things S04E01 4K'"""
    title_words = title.split()
    season = random.randint(1, 8)
    episode = random.randint(1, 24)
    se_marker = f"S{season:02d}E{episode:02d}"

    text = title_words + [se_marker] + quality_parts
    tags = (
        make_iob(title_words, "TITLE")
        + ["O"]
        + make_iob(quality_parts, "QUALITY")
    )
    return text, tags


def template_audio_classic(artist, track, quality):
    """
    Классический формат аудио: Artist Track Quality
    Пример: 'Queen Bohemian Rhapsody FLAC'
    """
    artist_words = artist.split()
    track_words = track.split()
    text = artist_words + track_words + [quality]
    tags = (
        make_iob(artist_words, "ARTIST")
        + make_iob(track_words, "TITLE")
        + make_iob([quality], "QUALITY")
    )
    return text, tags


def template_audio_no_quality(artist, track):
    """Аудио без качества: 'Pink Floyd Time'"""
    artist_words = artist.split()
    track_words = track.split()
    text = artist_words + track_words
    tags = (
        make_iob(artist_words, "ARTIST")
        + make_iob(track_words, "TITLE")
    )
    return text, tags


def template_audio_track_only(track):
    """Только название трека: 'Heroes'"""
    track_words = track.split()
    text = list(track_words)
    tags = make_iob(track_words, "TITLE")
    return text, tags


def template_audio_with_year(artist, track, year, quality):
    """С годом: 'Nirvana Smells Like Teen Spirit 1991 320kbps'"""
    artist_words = artist.split()
    track_words = track.split()
    text = artist_words + track_words + [year] + [quality]
    tags = (
        make_iob(artist_words, "ARTIST")
        + make_iob(track_words, "TITLE")
        + make_iob([year], "YEAR")
        + make_iob([quality], "QUALITY")
    )
    return text, tags


# =========================================================================
# ГЕНЕРАТОРЫ ОБРАЗЦОВ ПО КАТЕГОРИЯМ
# =========================================================================

def gen_movie_en():
    title = random.choice(MOVIES_EN)
    year = random.choice(YEARS)
    quality = random_video_quality()

    template = random.choices(
        [template_movie_classic, template_movie_year_first,
         template_movie_minimal, template_movie_no_year, template_movie_only_title],
        weights=[55, 10, 15, 15, 5],
        k=1
    )[0]

    if template == template_movie_classic:
        group = random.choice(RELEASE_GROUPS) if random.random() > 0.6 else None
        return template(title, year, quality, group)
    elif template == template_movie_minimal:
        return template(title, year)
    elif template == template_movie_no_year:
        return template(title, quality)
    elif template == template_movie_only_title:
        return template(title)
    else:
        return template(title, year, quality)


def gen_movie_ru_lat():
    title = random.choice(MOVIES_RU_LAT)
    year = random.choice(YEARS)
    quality = random_video_quality()
    template = random.choices(
        [template_movie_classic, template_movie_minimal, template_movie_no_year],
        weights=[60, 25, 15], k=1
    )[0]
    if template == template_movie_classic:
        return template(title, year, quality)
    elif template == template_movie_minimal:
        return template(title, year)
    else:
        return template(title, quality)


def gen_movie_ru_cyr():
    title = random.choice(MOVIES_RU_CYR)
    year = random.choice(YEARS)
    quality = random_video_quality()
    template = random.choices(
        [template_movie_classic, template_movie_minimal, template_movie_no_year],
        weights=[55, 30, 15], k=1
    )[0]
    if template == template_movie_classic:
        return template(title, year, quality)
    elif template == template_movie_minimal:
        return template(title, year)
    else:
        return template(title, quality)


def gen_tv_series():
    title = random.choice(TV_SERIES_EN + TV_SERIES_RU_LAT)
    year = random.choice(YEARS)
    quality = random_video_quality()
    if random.random() > 0.5:
        return template_tv_series(title, year, quality)
    else:
        return template_tv_series_no_year(title, quality)


def gen_audio_en():
    artist = random.choice(ARTISTS_EN)
    track = random.choice(TRACKS_EN)
    quality = random.choice(QUALITIES_AUDIO)
    year = random.choice(YEARS)

    template = random.choices(
        [template_audio_classic, template_audio_no_quality,
         template_audio_track_only, template_audio_with_year],
        weights=[55, 25, 5, 15], k=1
    )[0]

    if template == template_audio_classic:
        return template(artist, track, quality)
    elif template == template_audio_no_quality:
        return template(artist, track)
    elif template == template_audio_track_only:
        return template(track)
    else:
        return template(artist, track, year, quality)


def gen_audio_ru_lat():
    artist = random.choice(ARTISTS_RU_LAT)
    track = random.choice(TRACKS_RU_LAT)
    quality = random.choice(QUALITIES_AUDIO)
    template = random.choices(
        [template_audio_classic, template_audio_no_quality],
        weights=[70, 30], k=1
    )[0]
    if template == template_audio_classic:
        return template(artist, track, quality)
    else:
        return template(artist, track)


def gen_audio_ru_cyr():
    artist = random.choice(ARTISTS_RU_CYR)
    track = random.choice(TRACKS_RU_CYR)
    quality = random.choice(QUALITIES_AUDIO)
    template = random.choices(
        [template_audio_classic, template_audio_no_quality],
        weights=[65, 35], k=1
    )[0]
    if template == template_audio_classic:
        return template(artist, track, quality)
    else:
        return template(artist, track)


def gen_edge_case():
    """
    Генерирует короткие/неструктурированные имена.
    Эти примеры важны: модель должна научиться, что не каждая
    строка содержит все сущности, и что многие токены — это 'O'.
    """
    case = random.choice([
        "single_word", "year_only", "quality_only", "numbers_only",
        "tech_garbage", "two_words"
    ])

    if case == "single_word":
        # Одно случайное слово — никаких сущностей
        word = random.choice(["movie", "video", "audio", "test", "untitled",
                              "fragment", "clip", "rec", "scene"])
        return [word], ["O"]

    elif case == "year_only":
        year = random.choice(YEARS)
        return [year], ["B-YEAR"]

    elif case == "quality_only":
        q = random.choice(QUALITIES_VIDEO_RES + QUALITIES_AUDIO)
        return [q], ["B-QUALITY"]

    elif case == "numbers_only":
        # Случайный номер — это не год, это просто номер
        n = str(random.randint(1, 999))
        return [n], ["O"]

    elif case == "tech_garbage":
        # Случайный набор технических токенов без сущностей
        garbage = random.sample([
            "VIDEO", "TS", "DUMP", "RAW", "FINAL", "NEW", "OLD",
            "v1", "v2", "test", "draft", "backup", "copy"
        ], k=random.randint(1, 3))
        return garbage, ["O"] * len(garbage)

    else:  # two_words — два случайных слова
        words = random.sample([
            "video", "movie", "show", "demo", "rec", "session",
            "file", "media", "clip", "fragment"
        ], k=2)
        return words, ["O", "O"]


# Карта генераторов: ключ из SAMPLES_DISTRIBUTION → функция
GENERATORS = {
    "movie_en":     gen_movie_en,
    "movie_ru_lat": gen_movie_ru_lat,
    "movie_ru_cyr": gen_movie_ru_cyr,
    "tv_series":    gen_tv_series,
    "audio_en":     gen_audio_en,
    "audio_ru_lat": gen_audio_ru_lat,
    "audio_ru_cyr": gen_audio_ru_cyr,
    "edge_case":    gen_edge_case,
}


# =========================================================================
# MAIN
# =========================================================================

def main():
    random.seed(RANDOM_SEED)

    BASE_DIR = Path(__file__).resolve().parent.parent
    OUTPUT_FILE = BASE_DIR / "data" / "raw" / "synthetic_media_names.csv"
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    total = sum(SAMPLES_DISTRIBUTION.values())
    print("=" * 70)
    print(f"Генерация датасета: {OUTPUT_FILE}")
    print(f"Целевое количество строк: {total}")
    print(f"Random seed: {RANDOM_SEED}")
    print("=" * 70)

    rows = []
    for category, count in SAMPLES_DISTRIBUTION.items():
        gen_func = GENERATORS[category]
        category_rows = []
        for _ in range(count):
            text_words, tags = gen_func()

            # Sanity check: длины должны совпадать
            if len(text_words) != len(tags):
                continue

            text_str = " ".join(text_words)
            tags_str = ",".join(tags)
            category_rows.append((text_str, tags_str))

        rows.extend(category_rows)
        print(f"  [{category:15s}] сгенерировано: {len(category_rows)}")

    random.shuffle(rows)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "tags"])
        writer.writerows(rows)

    print("=" * 70)
    print(f"[OK] Записано в файл: {len(rows)} строк")

    # Статистика
    unique_texts = set(r[0] for r in rows)
    print(f"[OK] Уникальных текстов: {len(unique_texts)}")

    all_words = set()
    for text, _ in rows:
        all_words.update(text.split())
    print(f"[OK] Уникальных токенов в словаре: {len(all_words)}")

    print("=" * 70)


if __name__ == "__main__":
    main()