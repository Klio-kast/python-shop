"""
Management command: train_recommender
======================================
Обучает рекомендательную модель:
  1. TF-IDF словарь — на pelegelraz + товарах магазина (объединённый корпус).
  2. SVD — на pelegelraz.
  3. Индекс — проецирует товары магазина в пространство SVD.

Использование:
    python manage.py train_recommender

Требования:
    pip install scikit-learn datasets huggingface_hub
"""

import time
from django.core.management.base import BaseCommand, CommandError

from shop.recommender import (
    build_model, save_model, delete_model,
    MODEL_PATH, MODEL_VERSION,
)

PELEGELRAZ_ID = 'pelegelraz/perfumes-dataset'


class Command(BaseCommand):
    help = 'Обучает модель рекомендаций.'

    def handle(self, *args, **options):

        # ── Проверка scikit-learn ────────────────────────────────
        try:
            import sklearn
        except ImportError:
            raise CommandError(
                'Установите scikit-learn:\n'
                '  pip install scikit-learn --timeout 120'
            )

        # ── Диагностика БД ───────────────────────────────────────
        from shop.models import Product
        total = Product.objects.count()
        if total == 0:
            raise CommandError(
                'В БД нет товаров.\n'
                'Импортируйте: python manage.py import_mrbob'
            )

        # Показываем что реально есть в БД для диагностики
        # === Диагностика БД (только существующие поля) ===
        self.stdout.write("=== Диагностика БД ===")
        total = Product.objects.count()
        self.stdout.write(f"Всего товаров: {total}")

        if total > 0:
            sample = Product.objects.select_related('brand', 'category').first()
            self.stdout.write(f"Пример товара: {sample.name}")
            self.stdout.write(f"  brand     : {sample.brand}")
            self.stdout.write(f"  category  : {sample.category}")
            self.stdout.write(f"  volume    : {sample.volume} мл")
            self.stdout.write(f"  price     : {sample.price} ₽")
            self.stdout.write(f"  stock     : {sample.stock} шт")

            # Поля, которые реально есть в модели
            self.stdout.write(f"  top_notes    : {repr(str(sample.top_notes)[:80]) if sample.top_notes else '—'}")
            self.stdout.write(f"  middle_notes : {repr(str(sample.middle_notes)[:80]) if sample.middle_notes else '—'}")
            self.stdout.write(f"  base_notes   : {repr(str(sample.base_notes)[:80]) if sample.base_notes else '—'}")
            self.stdout.write(f"  main_accords : {repr(sample.main_accords) if sample.main_accords else '—'}")

        self.stdout.write("\nНачинаем обучение модели рекомендаций...\n")

        # === Заполненность полей (только те, что реально есть в модели) ===
        has_desc = Product.objects.exclude(description='').exclude(description__isnull=True).count()
        has_top = Product.objects.exclude(top_notes='').exclude(top_notes__isnull=True).count()
        has_mid = Product.objects.exclude(middle_notes='').exclude(middle_notes__isnull=True).count()
        has_base = Product.objects.exclude(base_notes='').exclude(base_notes__isnull=True).count()
        has_accords = Product.objects.exclude(main_accords={}).exclude(main_accords__isnull=True).count()
        has_cat = Product.objects.filter(category__isnull=False).count()

        self.stdout.write(f'\nЗаполненность полей:')
        self.stdout.write(f'  description   : {has_desc}/{total}')
        self.stdout.write(f'  top_notes     : {has_top}/{total}')
        self.stdout.write(f'  middle_notes  : {has_mid}/{total}')
        self.stdout.write(f'  base_notes    : {has_base}/{total}')
        self.stdout.write(f'  main_accords  : {has_accords}/{total}')
        self.stdout.write(f'  category      : {has_cat}/{total}')

        if has_desc == 0 and has_frag == 0 and has_ingr == 0:
            self.stdout.write(self.style.ERROR(
                '\n✗ Все текстовые поля пусты!\n'
                'Переимпортируйте товары:\n'
                '  python manage.py import_mrbob --flush'
            ))
            return

        # ── Скачиваем pelegelraz ─────────────────────────────────
        self.stdout.write(f'\n=== Загрузка датасета pelegelraz ===')
        try:
            from datasets import load_dataset
        except ImportError:
            raise CommandError('pip install datasets huggingface_hub')

        try:
            ds = load_dataset(PELEGELRAZ_ID)
            df = ds['train'].to_pandas()
        except Exception as e:
            raise CommandError(f'Ошибка загрузки датасета: {e}')

        df = df.dropna(subset=['all_notes'])
        df = df[df['all_notes'].str.strip() != '']
        self.stdout.write(self.style.SUCCESS(f'Загружено {len(df)} записей из pelegelraz.'))

        # ── Удаляем старую модель ────────────────────────────────
        if MODEL_PATH.exists():
            self.stdout.write(self.style.WARNING('\nУдаляю старую модель …'))
            delete_model()

        # ── Обучение ─────────────────────────────────────────────
        self.stdout.write('\n=== Обучение модели ===')
        t0 = time.time()

        try:
            model = build_model(df, verbose_callback=self.stdout.write)
        except ValueError as e:
            raise CommandError(str(e))
        except Exception as e:
            import traceback
            raise CommandError(f'Ошибка:\n{traceback.format_exc()}')

        elapsed = time.time() - t0

        # ── Сохранение ───────────────────────────────────────────
        save_model(model)

        # ── Итоговая диагностика ─────────────────────────────────
        import numpy as np
        norms = np.linalg.norm(model['shop_reduced_norm'], axis=1)
        nonzero = int(np.sum(norms > 1e-6))

        self.stdout.write(self.style.SUCCESS(
            f'\n=== Результат (версия модели v{MODEL_VERSION}) ===\n'
            f'  Товаров проиндексировано : {model["n_products"]}\n'
            f'  Товаров с ненулевым вектором: {nonzero}\n'
            f'  Словарь TF-IDF           : {model["vocab_size"]} токенов\n'
            f'  SVD компоненты           : {model["n_components"]}\n'
            f'  Время обучения           : {elapsed:.1f} сек\n'
            f'  Файл                     : {MODEL_PATH}\n'
        ))

        if nonzero == 0:
            self.stdout.write(self.style.ERROR(
                '✗ ВСЕ векторы нулевые — рекомендации не будут работать.\n'
                'Это означает, что поля товаров не пересекаются со словарём.\n'
                'Переимпортируйте товары: python manage.py import_mrbob --flush'
            ))
        elif nonzero < model['n_products']:
            self.stdout.write(self.style.WARNING(
                f'⚠ {model["n_products"] - nonzero} товаров с нулевым вектором.\n'
                'Эти товары не будут участвовать в рекомендациях.'
            ))
        else:
            self.stdout.write(self.style.SUCCESS(
                '✓ Все товары проиндексированы успешно!\n'
                'Откройте /recommend/ для проверки.'
            ))