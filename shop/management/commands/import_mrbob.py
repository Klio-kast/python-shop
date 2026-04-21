"""
Management command: import_mrbob
==================================
Импортирует парфюмы из датасета MrBob23/perfume-description в БД Django.

Датасет содержит 256 парфюмов с разбивкой нот на top/middle/base,
аккорды, сезонность, пол и изображения.

Использование:
    python manage.py import_mrbob

Требования:
    pip install huggingface_hub pandas
"""

import json
import pandas as pd
import random
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from shop.models import Brand, Category, Product


DATASET_ID = 'MrBob23/perfume-description'
FILENAME   = 'perfume_metadata.csv'


def _dominant_season(seasonal_ratings_json: str) -> str:
    """Возвращает доминирующий сезон по JSON рейтингов."""
    try:
        data = json.loads(seasonal_ratings_json)
        return max(data, key=data.get).capitalize()
    except Exception:
        return ''


def _parse_gender(gender_str: str) -> str:
    """Нормализует поле gender."""
    g = str(gender_str or '').lower().strip()
    if 'women' in g or 'female' in g or 'for her' in g:
        return 'Female'
    if 'men' in g or 'male' in g or 'for him' in g:
        return 'Male'
    if 'unisex' in g:
        return 'Unisex'
    return 'Unisex'


def _extract_brand(title: str) -> tuple[str, str]:
    """
    В датасете title = 'Perfume Name BrandName'.
    Бренд идёт последним словом/словами — попробуем разделить по известным брендам.
    Если не получается — берём последнее слово как бренд.
    Возвращает (brand_name, perfume_name).
    """
    # Fragrantica format: "Perfume Name Brand" — бренд последним
    # Но надёжнее всего — берём всё как название, бренд = 'Unknown'
    # В датасете есть url вида /perfume/BrandName/PerfumeName/ — парсим оттуда
    return 'Unknown', str(title or '').strip()


def _brand_from_url(url: str) -> str:
    """Извлекает бренд из URL fragrantica.com/perfume/Brand/Name/"""
    try:
        parts = str(url).rstrip('/').split('/')
        # .../perfume/Brand/Name/
        idx = parts.index('perfume')
        return parts[idx + 1].replace('-', ' ').title()
    except (ValueError, IndexError):
        return 'Unknown'


class Command(BaseCommand):
    help = 'Импортирует парфюмы из датасета MrBob23/perfume-description.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--flush', action='store_true',
            help='Удалить все товары перед импортом.',
        )
        parser.add_argument(
            '--local-csv', default=None,
            help='Путь к локальному perfume_metadata.csv вместо HuggingFace.',
        )

    def handle(self, *args, **options):
        # ── Загрузка данных ──────────────────────────────────────────
        local_csv = options.get('local_csv')

        if local_csv:
            self.stdout.write(f'Читаю локальный CSV: {local_csv}')
            df = pd.read_csv(local_csv)
        else:
            self.stdout.write(f'Скачиваю {DATASET_ID}/{FILENAME} …')
            try:
                from huggingface_hub import hf_hub_download
            except ImportError:
                raise CommandError('pip install huggingface_hub')
            try:
                path = hf_hub_download(
                    repo_id=DATASET_ID, repo_type='dataset', filename=FILENAME
                )
                df = pd.read_csv(path)
            except Exception as e:
                raise CommandError(f'Ошибка загрузки: {e}')

        self.stdout.write(self.style.SUCCESS(f'Загружено {len(df)} записей.'))

        # ── Очистка ──────────────────────────────────────────────────
        if options['flush']:
            self.stdout.write(self.style.WARNING('Удаляю все товары …'))
            Product.objects.all().delete()
            Brand.objects.all().delete()
            Category.objects.all().delete()

        # ── Импорт ───────────────────────────────────────────────────
        created = skipped = errors = 0
        brand_cache: dict[str, Brand] = {}
        cat_cache:   dict[str, Category] = {}

        # Семейство для MrBob23 — нет поля family, используем main_accords
        def get_family(row) -> str:
            try:
                accords = json.loads(row.get('main_accords', '{}') or '{}')
                if accords:
                    return max(accords, key=accords.get).title()
            except Exception:
                pass
            return 'Other'

        with transaction.atomic():
            for _, row in df.iterrows():
                try:
                    title = str(row.get('title', '') or '').strip()
                    if not title:
                        skipped += 1
                        continue

                    # Бренд из URL
                    brand_name = _brand_from_url(str(row.get('url', '')))
                    if brand_name not in brand_cache:
                        b, _ = Brand.objects.get_or_create(name=brand_name)
                        brand_cache[brand_name] = b
                    brand_obj = brand_cache[brand_name]

                    # Категория = доминирующий аккорд
                    family = get_family(row)
                    if family not in cat_cache:
                        c, _ = Category.objects.get_or_create(name=family)
                        cat_cache[family] = c
                    cat_obj = cat_cache[family]

                    # Пропустить существующие
                    if Product.objects.filter(name=title, brand=brand_obj).exists():
                        skipped += 1
                        continue

                    # Все ноты объединяем в ingredients
                    top    = str(row.get('top_notes',    '') or '')
                    middle = str(row.get('middle_notes', '') or '')
                    base   = str(row.get('base_notes',   '') or '')
                    all_notes_parts = [n for n in [top, middle, base] if n.strip()]
                    ingredients = ', '.join(all_notes_parts)

                    # Аккорды → fragrances
                    try:
                        accords_dict = json.loads(row.get('main_accords', '{}') or '{}')
                        # Берём топ-5 аккордов по весу
                        top_accords = sorted(accords_dict.items(), key=lambda x: -x[1])[:5]
                        fragrances = ', '.join(k for k, _ in top_accords)
                    except Exception:
                        fragrances = ''

                    # Сезон как подсемейство
                    subfamily = _dominant_season(str(row.get('seasonal_ratings', '') or ''))

                    # Изображение
                    image_url = str(row.get('image_url', '') or '').strip()

                    # === ИСПРАВЛЕННЫЙ БЛОК СОЗДАНИЯ ПРОДУКТА ===
                    # Используем только те поля, которые реально есть в модели Product
                    product = Product(
                        name=title[:200],
                        brand=brand_obj,
                        category=cat_obj,
                        volume=random.randint(30, 200),  # случайный объём
                        description=str(row.get('description', '') or '').strip(),
                        price=round(random.uniform(49.99, 349.99), 2),
                        stock=random.randint(5, 50),

                        # Поля, которые есть в вашей модели:
                        top_notes=str(row.get('top_notes', '') or '').strip(),
                        middle_notes=str(row.get('middle_notes', '') or '').strip(),
                        base_notes=str(row.get('base_notes', '') or '').strip(),

                        # main_accords — JSONField, сохраняем как есть
                        main_accords=row.get('main_accords', {}),

                        # image_url (если у вас есть такое поле)
                        image_url=str(row.get('image_url', '') or '').strip(),
                    )

                    product.save()
                    created += 1

                except Exception as e:
                    errors += 1
                    self.stderr.write(self.style.ERROR(f'Ошибка: {row.get("title","?")} — {e}'))
                    if errors > 50:
                        raise CommandError('Слишком много ошибок, импорт прерван.')

        self.stdout.write(self.style.SUCCESS(
            f'\n✓ Готово!\n'
            f'  Создано  : {created}\n'
            f'  Пропущено: {skipped}\n'
            f'  Ошибок   : {errors}\n'
        ))
        self.stdout.write('Следующий шаг: python manage.py train_recommender')