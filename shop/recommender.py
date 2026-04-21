"""
shop/recommender.py
====================
Архитектура:

  ОБУЧЕНИЕ TF-IDF:
    Словарь строится на ОБЪЕДИНЕНИИ pelegelraz + текстов товаров магазина.
    Это гарантирует, что все слова из описаний/нот магазина попадут в словарь.

  ИНДЕКСАЦИЯ МАГАЗИНА:
    Каждый товар магазина превращается в вектор через обученный TF-IDF + SVD.
    Используются ВСЕ доступные текстовые поля: description, fragrances,
    ingredients, category.name, subfamily, brand.name.

  ПОИСК:
    Запрос → TF-IDF → SVD → cosine similarity со всеми товарами магазина.
    Возвращаются pk реальных Product из БД.
"""

import pickle
import numpy as np
from pathlib import Path

MODEL_DIR     = Path(__file__).resolve().parent.parent / 'ml_models'
MODEL_PATH    = MODEL_DIR / 'recommender_model.pkl'
MODEL_VERSION = 5  # увеличиваем при изменении схемы


# ─────────────────────────────────────────────────────────────────
# Сохранение / загрузка
# ─────────────────────────────────────────────────────────────────

def save_model(obj: dict):
    obj['_version'] = MODEL_VERSION
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(obj, f)


def load_model() -> dict | None:
    """None — если файл отсутствует, повреждён или версия устарела."""
    if not MODEL_PATH.exists():
        return None
    try:
        with open(MODEL_PATH, 'rb') as f:
            obj = pickle.load(f)
        required = {'vectorizer', 'svd', 'shop_reduced_norm', 'product_pks', '_version'}
        if not required.issubset(obj.keys()):
            return None
        if obj.get('_version') != MODEL_VERSION:
            return None
        return obj
    except Exception:
        return None


def delete_model():
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()


# ─────────────────────────────────────────────────────────────────
# Формирование признаковых строк
# ─────────────────────────────────────────────────────────────────

def _shop_product_feature_string(product) -> str:
    """
    Использует ВСЕ доступные текстовые поля товара магазина.
    Поля с нотами повторяются для увеличения веса.
    """
    parts = []

    # 1. description — содержит текст с упоминанием нот (×3, главный сигнал)
    desc = str(getattr(product, 'description', '') or '').strip()
    if desc:
        parts += [desc] * 3

    # 2. fragrances — топ-аккорды (woody, sweet, …) (×3)
    fragrances = str(getattr(product, 'fragrances', '') or '').strip()
    if fragrances:
        parts += [fragrances.replace(',', ' ')] * 3

    # 3. ingredients — top+middle+base notes объединённые (×2)
    ingredients = getattr(product, 'ingredients', None)
    if isinstance(ingredients, list):
        notes = ' '.join(str(i) for i in ingredients if i)
    elif isinstance(ingredients, str):
        notes = ingredients.replace(',', ' ').strip()
    else:
        notes = ''
    if notes:
        parts += [notes] * 2

    # 4. category.name — доминирующий аккорд/семейство (×2)
    try:
        cat = product.category.name.replace(' ', '_')
        parts += [cat] * 2
    except Exception:
        pass

    # 5. subfamily — доминирующий сезон (×1)
    sub = str(getattr(product, 'subfamily', '') or '').strip()
    if sub:
        parts.append(sub.replace(' ', '_'))

    # 6. gender (×1)
    gender = str(getattr(product, 'gender', '') or '').strip()
    if gender:
        parts.append(gender)

    return ' '.join(parts)


def _pelegelraz_feature_string(row: dict) -> str:
    """Признаковая строка из записи pelegelraz."""
    parts = []

    all_notes = str(row.get('all_notes', '') or '').strip()
    if all_notes:
        parts += [all_notes.replace(',', ' ')] * 3

    base = str(row.get('base_notes', '') or '').strip()
    if base:
        parts += [base.replace(',', ' ')] * 2

    family = str(row.get('family', '') or '').strip().replace(' ', '_')
    if family:
        parts += [family] * 2

    occasions = str(row.get('occasions', '') or '').strip()
    if occasions:
        parts.append(occasions.replace(',', ' '))

    moods = str(row.get('moods', '') or '').strip()
    if moods:
        parts.append(moods.replace(',', ' '))

    desc = str(row.get('professional_description', '') or '').strip()
    if desc:
        parts.append(desc)

    return ' '.join(parts)


# ─────────────────────────────────────────────────────────────────
# Построение модели
# ─────────────────────────────────────────────────────────────────

def build_model(pelegelraz_df, verbose_callback=None) -> dict:
    """
    Строит модель рекомендаций.

    pelegelraz_df — pandas DataFrame датасета pelegelraz.
    verbose_callback — функция для вывода сообщений (например, self.stdout.write).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize
    from shop.models import Product

    def log(msg):
        if verbose_callback:
            verbose_callback(msg)

    # ── Шаг 1: подготовка товаров магазина ──────────────────────
    log('  Загружаю товары магазина из БД …')
    shop_products = list(
        Product.objects.select_related('brand', 'category').order_by('id')
    )
    if not shop_products:
        raise ValueError('В БД нет товаров. Запустите: python manage.py import_mrbob')

    shop_strings = []
    product_pks  = []
    empty_count  = 0

    for p in shop_products:
        fs = _shop_product_feature_string(p)
        if fs.strip():
            shop_strings.append(fs)
            product_pks.append(p.pk)
        else:
            empty_count += 1

    log(f'  Товаров с признаками : {len(product_pks)} из {len(shop_products)}')
    if empty_count > 0:
        log(f'  Товаров без признаков: {empty_count} (будут пропущены)')

    if not product_pks:
        raise ValueError(
            'Ни у одного товара нет текстовых полей.\n'
            'Переимпортируйте: python manage.py import_mrbob --flush'
        )

    # ── Шаг 2: признаки pelegelraz ──────────────────────────────
    log('  Формирую признаки pelegelraz …')
    pelegelraz_strings = [
        _pelegelraz_feature_string(row)
        for row in pelegelraz_df.to_dict('records')
    ]
    pelegelraz_strings = [s for s in pelegelraz_strings if s.strip()]
    log(f'  Записей pelegelraz   : {len(pelegelraz_strings)}')

    # ── Шаг 3: обучаем TF-IDF на ОБЪЕДИНЁННОМ корпусе ──────────
    # Это гарантирует, что все слова из товаров магазина попадут в словарь
    log('  Обучаю TF-IDF на объединённом корпусе …')
    combined_corpus = pelegelraz_strings + shop_strings

    vectorizer = TfidfVectorizer(
        analyzer='word',
        token_pattern=r'[a-zA-Z][a-zA-Z0-9_]*',
        ngram_range=(1, 2),
        min_df=1,
        max_features=10000,
        sublinear_tf=True,      # сглаживает влияние частых слов
    )
    vectorizer.fit(combined_corpus)
    log(f'  Словарь TF-IDF       : {len(vectorizer.vocabulary_)} токенов')

    # ── Шаг 4: обучаем SVD на pelegelraz ────────────────────────
    log('  Обучаю SVD на pelegelraz …')
    pelegelraz_tfidf = vectorizer.transform(pelegelraz_strings)
    n_components = min(100, pelegelraz_tfidf.shape[0] - 1, pelegelraz_tfidf.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(pelegelraz_tfidf)
    log(f'  SVD компоненты       : {n_components}')

    # ── Шаг 5: проецируем товары магазина ───────────────────────
    log('  Проецирую товары магазина в пространство SVD …')
    shop_tfidf   = vectorizer.transform(shop_strings)
    shop_reduced = svd.transform(shop_tfidf)

    # Проверяем на нулевые векторы
    norms = np.linalg.norm(shop_reduced, axis=1)
    zero_vectors = int(np.sum(norms < 1e-10))
    if zero_vectors > 0:
        log(f'  ⚠ Нулевых векторов: {zero_vectors} — у этих товаров нет совпадений со словарём')

    shop_norm = normalize(shop_reduced, norm='l2')

    # Заменяем NaN (от нулевых векторов) на 0
    shop_norm = np.nan_to_num(shop_norm, nan=0.0)

    log(f'  Проиндексировано     : {len(product_pks)} товаров')

    return {
        'vectorizer':        vectorizer,
        'svd':               svd,
        'shop_reduced_norm': shop_norm,
        'product_pks':       product_pks,
        'n_products':        len(product_pks),
        'vocab_size':        len(vectorizer.vocabulary_),
        'n_components':      n_components,
    }


# ─────────────────────────────────────────────────────────────────
# Поиск
# ─────────────────────────────────────────────────────────────────

def get_similar_pks(product_pk: int, model: dict, top_n: int = 6) -> list[tuple[int, float]]:
    """Item-to-item: возвращает [(pk, score), …] top_n похожих товаров."""
    pks       = model['product_pks']
    shop_norm = model['shop_reduced_norm']

    if product_pk not in pks:
        return []

    idx    = pks.index(product_pk)
    vec    = shop_norm[idx].reshape(1, -1)
    scores = (shop_norm @ vec.T).flatten()
    scores[idx] = -1

    top_idx = np.argsort(scores)[::-1][:top_n]
    return [(pks[i], float(scores[i])) for i in top_idx]


def get_pks_by_query(query: str, model: dict, top_n: int = 12) -> list[tuple[int, float]]:
    """Поиск по запросу пользователя среди товаров магазина."""
    from sklearn.preprocessing import normalize

    vectorizer = model['vectorizer']
    svd        = model['svd']
    shop_norm  = model['shop_reduced_norm']
    pks        = model['product_pks']

    q_tfidf   = vectorizer.transform([query])
    q_reduced = svd.transform(q_tfidf)
    q_norm    = normalize(q_reduced, norm='l2')
    q_norm    = np.nan_to_num(q_norm, nan=0.0)

    scores  = (shop_norm @ q_norm.T).flatten()
    top_idx = np.argsort(scores)[::-1][:top_n]
    return [(pks[i], float(scores[i])) for i in top_idx]