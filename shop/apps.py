"""
Configuration for the shop application.

Автоматический импорт датасета MrBob23/perfume-description при первом запуске runserver.
"""
from django.apps import AppConfig
import sys
import os

class ShopConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'shop'

    def ready(self):
        """
        Выполняется при старте приложения.
        Автоматически запускает импорт датасета только один раз.
        """
        # Выполняем только при запуске сервера разработки
        if 'runserver' not in sys.argv and 'runserver_plus' not in sys.argv:
            return

        from shop.models import Product

        # Проверяем, был ли уже импорт (файл-флаг)
        flag_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'import_done.flag')

        if os.path.exists(flag_file):
            return  # Импорт уже выполнен ранее

        # Если товаров в базе ещё нет — запускаем импорт
        if Product.objects.count() == 0:
            print("\n🚀 Запускается автоматический импорт датасета MrBob23/perfume-description...\n")

            try:
                from django.core.management import call_command
                # Запускаем команду импорта
                call_command('import_perfume_dataset', 'perfume_metadata.csv', verbosity=1)

                # Создаём файл-флаг, чтобы импорт больше не повторялся
                with open(flag_file, 'w') as f:
                    f.write('import completed')

                print("✅ Импорт датасета успешно завершён! Товары и изображения загружены.\n")
            except Exception as e:
                print(f"❌ Ошибка при импорте датасета: {e}")
