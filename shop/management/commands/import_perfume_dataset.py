"""
Management command для импорта датасета MrBob23/perfume-description
с автоматической загрузкой изображений.
"""
import os
import requests
from django.core.files.base import ContentFile
from django.core.management.base import BaseCommand
import pandas as pd
from shop.models import Brand, Category, Product
from decimal import Decimal
import random

class Command(BaseCommand):
    help = 'Импорт датасета MrBob23/perfume-description с загрузкой изображений'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str, help='Путь к perfume_metadata.csv')

    def handle(self, *args, **options):
        csv_path = options['csv_file']

        if not os.path.exists(csv_path):
            self.stderr.write(self.style.ERROR(f'Файл не найден: {csv_path}'))
            return

        self.stdout.write(self.style.SUCCESS('Начинаем импорт датасета MrBob23/perfume-description...'))

        df = pd.read_csv(csv_path)
        created_count = 0

        for _, row in df.iterrows():
            title = str(row.get('title', '')).strip()
            if not title:
                continue

            # Пропускаем, если товар уже существует
            if Product.objects.filter(name=title).exists():
                continue

            # Создаём бренд и категорию
            brand_name = title.split()[0] if ' ' in title else title
            brand, _ = Brand.objects.get_or_create(name=brand_name)

            category_name = row.get('gender', 'Unisex')
            category, _ = Category.objects.get_or_create(name=category_name)

            # Создаём продукт
            product = Product.objects.create(
                name=title,
                brand=brand,
                category=category,
                description=row.get('description', ''),
                top_notes=row.get('top_notes', ''),
                middle_notes=row.get('middle_notes', ''),
                base_notes=row.get('base_notes', ''),
                main_accords=eval(row['main_accords']) if pd.notna(row.get('main_accords')) else {},
                gender_ratings=eval(row['gender_ratings']) if pd.notna(row.get('gender_ratings')) else {},
                seasonal_ratings=eval(row['seasonal_ratings']) if pd.notna(row.get('seasonal_ratings')) else {},
                price=Decimal(random.uniform(60, 250)),
                stock=random.randint(10, 100),
                image_url=row.get('image_url', None),
            )

            # Автоматическая загрузка изображения
            image_url = row.get('image_url')
            if image_url and str(image_url).startswith(('http://', 'https://')):
                try:
                    r = requests.get(image_url, timeout=15)
                    if r.status_code == 200:
                        filename = os.path.basename(image_url) or f"perfume_{product.id}.jpg"
                        product.image.save(filename, ContentFile(r.content), save=True)
                        self.stdout.write(f'✓ Изображение загружено: {filename}')
                except Exception as e:
                    self.stderr.write(f'✗ Не удалось скачать изображение для {title}: {e}')

            created_count += 1

        self.stdout.write(self.style.SUCCESS(
            f'Импорт завершён! Добавлено товаров: {created_count}'
        ))