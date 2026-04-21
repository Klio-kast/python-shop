"""
Management command: test_recommender
======================================
Запускает все тесты рекомендательной модели и сохраняет результаты
в файл test_results/recommender_test_results.json

Использование:
    python manage.py test_recommender

После выполнения запустите визуализацию:
    python visualize_tests.py
"""

import json
import time
import numpy as np
import pickle
from pathlib import Path
from django.core.management.base import BaseCommand, CommandError


RESULTS_DIR = Path('test_results')
RESULTS_FILE = RESULTS_DIR / 'recommender_test_results.json'


class Command(BaseCommand):
    help = 'Тестирует рекомендательную модель и сохраняет результаты.'

    def handle(self, *args, **options):
        RESULTS_DIR.mkdir(exist_ok=True)

        self.stdout.write('\n' + '='*60)
        self.stdout.write('  ТЕСТИРОВАНИЕ РЕКОМЕНДАТЕЛЬНОЙ МОДЕЛИ')
        self.stdout.write('='*60 + '\n')

        # Загрузка модели
        try:
            from shop.recommender import load_model, get_pks_by_query, get_similar_pks
            from shop.models import Product
        except ImportError as e:
            raise CommandError(f'Ошибка импорта: {e}')

        model = load_model()
        if model is None:
            raise CommandError(
                'Модель не обучена или устарела.\n'
                'Запустите: python manage.py train_recommender'
            )

        all_results = {}

        # ── ТЕСТ 1: Ненулевые векторы ────────────────────────────
        self.stdout.write('► Тест 1: Проверка векторов товаров ...')
        t1 = self._test_vectors(model)
        all_results['test1_vectors'] = t1
        status = self.style.SUCCESS('✓ ПРОЙДЕН') if t1['passed'] else self.style.ERROR('✗ ПРОВАЛЕН')
        self.stdout.write(f'  {status}')
        self.stdout.write(f'  Ненулевых векторов: {t1["nonzero"]} из {t1["total"]}')
        self.stdout.write(f'  Средняя норма: {t1["mean_norm"]:.4f}\n')

        # ── ТЕСТ 2: Релевантность запросов ───────────────────────
        self.stdout.write('► Тест 2: Релевантность результатов поиска ...')
        t2 = self._test_relevance(model, get_pks_by_query)
        all_results['test2_relevance'] = t2
        for q in t2['queries']:
            match_pct = q['category_match_pct']
            mark = '✓' if match_pct >= 50 else '~' if match_pct >= 30 else '✗'
            self.stdout.write(
                f'  {mark} «{q["query"]}» → '
                f'совпадение категорий: {match_pct:.0f}%, '
                f'топ-схожесть: {q["top_score"]:.1f}%'
            )
        self.stdout.write()

        # ── ТЕСТ 3: Различие результатов ─────────────────────────
        self.stdout.write('► Тест 3: Различие результатов для разных запросов ...')
        t3 = self._test_diversity(model, get_pks_by_query)
        all_results['test3_diversity'] = t3
        status = self.style.SUCCESS('✓ ПРОЙДЕН') if t3['passed'] else self.style.ERROR('✗ ПРОВАЛЕН')
        self.stdout.write(f'  {status}')
        for pair in t3['pairs']:
            self.stdout.write(
                f'  «{pair["q1"]}» vs «{pair["q2"]}»: '
                f'совпадений {pair["overlap"]} из 5 '
                f'(уникальных: {pair["unique_pct"]:.0f}%)'
            )
        self.stdout.write()

        # ── ТЕСТ 4: Item-to-item рекомендации ────────────────────
        self.stdout.write('► Тест 4: Item-to-item рекомендации ...')
        t4 = self._test_item_to_item(model, get_similar_pks)
        all_results['test4_item_to_item'] = t4
        for item in t4['items']:
            match_pct = item['category_match_pct']
            mark = '✓' if match_pct >= 50 else '~'
            self.stdout.write(
                f'  {mark} «{item["product_name"]}» [{item["category"]}]: '
                f'совпадение категорий {match_pct:.0f}%, '
                f'средняя схожесть {item["mean_score"]:.1f}%'
            )
        self.stdout.write()

        # ── ТЕСТ 5: Граничные случаи ─────────────────────────────
        self.stdout.write('► Тест 5: Граничные случаи ...')
        t5 = self._test_edge_cases(model, get_pks_by_query)
        all_results['test5_edge_cases'] = t5
        for case in t5['cases']:
            mark = '✓' if case['passed'] else '✗'
            self.stdout.write(f'  {mark} {case["name"]}: {case["description"]}')
        self.stdout.write()

        # ── ТЕСТ 6: Производительность ───────────────────────────
        self.stdout.write('► Тест 6: Производительность (100 итераций) ...')
        t6 = self._test_performance(model, get_pks_by_query, get_similar_pks)
        all_results['test6_performance'] = t6
        self.stdout.write(f'  Загрузка модели     : {t6["load_time_ms"]:.1f} мс')
        self.stdout.write(f'  Поиск по запросу    : {t6["query_avg_ms"]:.2f} мс (среднее)')
        self.stdout.write(f'  Item-to-item поиск  : {t6["item_avg_ms"]:.2f} мс (среднее)')
        perf_ok = t6["query_avg_ms"] < 50 and t6["item_avg_ms"] < 20
        status = self.style.SUCCESS('✓ ПРОЙДЕН') if perf_ok else self.style.WARNING('~ ПРИЕМЛЕМО')
        self.stdout.write(f'  {status}\n')

        # ── Итоговая сводка ───────────────────────────────────────
        self.stdout.write('='*60)
        passed = sum([
            t1['passed'],
            t2['overall_passed'],
            t3['passed'],
            t4['overall_passed'],
            all(c['passed'] for c in t5['cases']),
            perf_ok,
        ])
        total_tests = 6
        self.stdout.write(f'  ИТОГО: {passed}/{total_tests} тестов пройдено')
        self.stdout.write('='*60 + '\n')

        # Сохраняем результаты
        all_results['summary'] = {
            'passed': passed,
            'total': total_tests,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_products': model['n_products'],
            'vocab_size': model['vocab_size'],
            'n_components': model['n_components'],
        }

        with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        self.stdout.write(self.style.SUCCESS(
            f'Результаты сохранены в {RESULTS_FILE}\n'
            f'Для построения графиков запустите:\n'
            f'  python visualize_tests.py'
        ))

    # ─────────────────────────────────────────────────────────────
    # Тест 1: Ненулевые векторы
    # ─────────────────────────────────────────────────────────────
    def _test_vectors(self, model):
        matrix = model['shop_reduced_norm']
        norms = np.linalg.norm(matrix, axis=1)

        total    = len(norms)
        nonzero  = int(np.sum(norms > 1e-6))
        zero     = total - nonzero
        mean_norm = float(np.mean(norms))
        min_norm  = float(np.min(norms))
        max_norm  = float(np.max(norms))

        # Гистограмма норм по корзинам
        bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
        hist, _ = np.histogram(norms, bins=bins)
        norm_distribution = {
            f'{bins[i]:.1f}–{bins[i+1]:.1f}': int(hist[i])
            for i in range(len(hist))
        }

        return {
            'total': total,
            'nonzero': nonzero,
            'zero': zero,
            'mean_norm': mean_norm,
            'min_norm': min_norm,
            'max_norm': max_norm,
            'norm_distribution': norm_distribution,
            'passed': nonzero > total * 0.9,  # 90%+ ненулевых
        }

    # ─────────────────────────────────────────────────────────────
    # Тест 2: Релевантность запросов
    # ─────────────────────────────────────────────────────────────
    def _test_relevance(self, model, get_pks_by_query):
        from shop.models import Product

        test_cases = [
            {
                'query': 'jasmine rose peony',
                'expected_families': ['floral', 'flora', 'flower'],
            },
            {
                'query': 'woody cedar vetiver patchouli',
                'expected_families': ['woody', 'wood', 'chypre'],
            },
            {
                'query': 'vanilla amber musk sweet',
                'expected_families': ['oriental', 'amber', 'gourmand', 'sweet'],
            },
            {
                'query': 'citrus fresh bergamot lemon',
                'expected_families': ['citrus', 'fresh', 'aromatic', 'green'],
            },
            {
                'query': 'oud incense resin dark',
                'expected_families': ['oud', 'oriental', 'amber', 'woody'],
            },
        ]

        results = []
        all_passed = 0

        for case in test_cases:
            pairs = get_pks_by_query(case['query'], model, top_n=6)
            pk_list = [pk for pk, _ in pairs]
            score_map = {pk: s for pk, s in pairs}

            products = list(
                Product.objects.select_related('category')
                .filter(pk__in=pk_list)
            )

            top_score = round(score_map[pk_list[0]] * 100, 1) if pk_list else 0

            # Считаем совпадение категорий (нечёткое)
            matches = 0
            categories_found = []
            for p in products:
                cat = (p.category.name if p.category_id else '').lower()
                categories_found.append(cat)
                if any(exp in cat for exp in case['expected_families']):
                    matches += 1

            match_pct = (matches / len(products) * 100) if products else 0
            passed = match_pct >= 30 or top_score >= 5

            results.append({
                'query': case['query'],
                'expected_families': case['expected_families'],
                'top_score': top_score,
                'scores': [round(score_map[pk] * 100, 1) for pk in pk_list],
                'categories_found': list(set(categories_found)),
                'category_match_pct': round(match_pct, 1),
                'n_results': len(products),
                'passed': passed,
            })
            if passed:
                all_passed += 1

        return {
            'queries': results,
            'overall_passed': all_passed >= 3,
            'passed_count': all_passed,
        }

    # ─────────────────────────────────────────────────────────────
    # Тест 3: Различие результатов
    # ─────────────────────────────────────────────────────────────
    def _test_diversity(self, model, get_pks_by_query):
        query_pairs = [
            ('jasmine rose floral', 'woody cedar smoky'),
            ('vanilla sweet amber', 'fresh citrus bergamot'),
            ('oud dark oriental', 'aquatic marine clean'),
        ]

        pairs_results = []
        all_passed = 0

        for q1, q2 in query_pairs:
            r1 = [pk for pk, _ in get_pks_by_query(q1, model, top_n=5)]
            r2 = [pk for pk, _ in get_pks_by_query(q2, model, top_n=5)]
            overlap = len(set(r1) & set(r2))
            unique_pct = (1 - overlap / 5) * 100
            passed = overlap <= 2  # не более 2 совпадений из 5

            pairs_results.append({
                'q1': q1,
                'q2': q2,
                'pks1': r1,
                'pks2': r2,
                'overlap': overlap,
                'unique_pct': round(unique_pct, 1),
                'passed': passed,
            })
            if passed:
                all_passed += 1

        return {
            'pairs': pairs_results,
            'passed': all_passed >= 2,
        }

    # ─────────────────────────────────────────────────────────────
    # Тест 4: Item-to-item
    # ─────────────────────────────────────────────────────────────
    def _test_item_to_item(self, model, get_similar_pks):
        from shop.models import Product

        # Берём 5 разных товаров из разных категорий
        products = list(
            Product.objects.select_related('category')
            .order_by('category', 'id')
            .distinct()[:5]
        )

        items_results = []
        passed_count = 0

        for product in products:
            pairs = get_similar_pks(product.pk, model, top_n=6)
            pk_list = [pk for pk, _ in pairs]
            score_map = {pk: s for pk, s in pairs}

            similar = list(
                Product.objects.select_related('category')
                .filter(pk__in=pk_list)
            )

            source_cat = (product.category.name if product.category_id else '').lower()
            matches = sum(
                1 for p in similar
                if (p.category.name if p.category_id else '').lower() == source_cat
            )
            match_pct = (matches / len(similar) * 100) if similar else 0
            scores = [round(score_map[pk] * 100, 1) for pk in pk_list if pk in score_map]
            mean_score = sum(scores) / len(scores) if scores else 0
            passed = mean_score > 0 and len(similar) > 0

            items_results.append({
                'product_name': product.name[:40],
                'product_pk': product.pk,
                'category': source_cat,
                'n_similar': len(similar),
                'scores': scores,
                'mean_score': round(mean_score, 1),
                'category_match_pct': round(match_pct, 1),
                'passed': passed,
            })
            if passed:
                passed_count += 1

        return {
            'items': items_results,
            'overall_passed': passed_count >= 3,
            'passed_count': passed_count,
        }

    # ─────────────────────────────────────────────────────────────
    # Тест 5: Граничные случаи
    # ─────────────────────────────────────────────────────────────
    def _test_edge_cases(self, model, get_pks_by_query):
        cases = []

        # 1. Несуществующее слово
        try:
            result = get_pks_by_query('xyzqwerty12345abc', model, top_n=5)
            scores = [s for _, s in result]
            max_score = max(scores) if scores else 0
            cases.append({
                'name': 'Несуществующее слово',
                'input': 'xyzqwerty12345abc',
                'max_score': round(max_score * 100, 2),
                'n_results': len(result),
                'passed': True,
                'description': f'Вернул {len(result)} результатов, макс. схожесть {round(max_score*100,2)}%',
            })
        except Exception as e:
            cases.append({'name': 'Несуществующее слово', 'passed': False,
                          'description': f'ОШИБКА: {e}', 'max_score': 0, 'n_results': 0, 'input': ''})

        # 2. Кириллица
        try:
            result = get_pks_by_query('жасмин роза бергамот', model, top_n=5)
            cases.append({
                'name': 'Запрос на кириллице',
                'input': 'жасмин роза бергамот',
                'max_score': 0,
                'n_results': len(result),
                'passed': True,
                'description': f'Без ошибок, вернул {len(result)} результатов',
            })
        except Exception as e:
            cases.append({'name': 'Запрос на кириллице', 'passed': False,
                          'description': f'ОШИБКА: {e}', 'max_score': 0, 'n_results': 0, 'input': ''})

        # 3. Очень длинный запрос
        long_query = 'jasmine ' * 50
        try:
            result = get_pks_by_query(long_query.strip(), model, top_n=5)
            cases.append({
                'name': 'Очень длинный запрос',
                'input': 'jasmine × 50',
                'max_score': round(max(s for _, s in result) * 100, 2) if result else 0,
                'n_results': len(result),
                'passed': True,
                'description': f'Без ошибок, вернул {len(result)} результатов',
            })
        except Exception as e:
            cases.append({'name': 'Очень длинный запрос', 'passed': False,
                          'description': f'ОШИБКА: {e}', 'max_score': 0, 'n_results': 0, 'input': ''})

        # 4. Пустая строка
        try:
            result = get_pks_by_query('   ', model, top_n=5)
            cases.append({
                'name': 'Пустая строка (пробелы)',
                'input': '   ',
                'max_score': 0,
                'n_results': len(result),
                'passed': True,
                'description': f'Без ошибок, вернул {len(result)} результатов',
            })
        except Exception as e:
            cases.append({'name': 'Пустая строка (пробелы)', 'passed': False,
                          'description': f'ОШИБКА: {e}', 'max_score': 0, 'n_results': 0, 'input': ''})

        # 5. Одно известное слово
        try:
            result = get_pks_by_query('jasmine', model, top_n=5)
            scores = [s for _, s in result]
            max_score = max(scores) if scores else 0
            cases.append({
                'name': 'Одно ключевое слово',
                'input': 'jasmine',
                'max_score': round(max_score * 100, 2),
                'n_results': len(result),
                'passed': max_score > 0,
                'description': f'Вернул {len(result)} результатов, макс. схожесть {round(max_score*100,2)}%',
            })
        except Exception as e:
            cases.append({'name': 'Одно ключевое слово', 'passed': False,
                          'description': f'ОШИБКА: {e}', 'max_score': 0, 'n_results': 0, 'input': ''})

        return {'cases': cases}

    # ─────────────────────────────────────────────────────────────
    # Тест 6: Производительность
    # ─────────────────────────────────────────────────────────────
    def _test_performance(self, model, get_pks_by_query, get_similar_pks):
        from shop.models import Product
        from shop.recommender import MODEL_PATH

        # Загрузка модели
        t0 = time.time()
        from shop.recommender import load_model
        load_model()
        load_time_ms = (time.time() - t0) * 1000

        # Поиск по запросу — 100 итераций
        queries = [
            'jasmine rose bergamot',
            'woody cedar vetiver',
            'vanilla amber musk',
            'fresh citrus lemon',
            'oud incense resin',
        ]
        times_query = []
        for q in queries * 20:  # 100 итераций
            t0 = time.time()
            get_pks_by_query(q, model, top_n=12)
            times_query.append((time.time() - t0) * 1000)

        # Item-to-item — 100 итераций
        pks = list(Product.objects.values_list('pk', flat=True)[:20])
        times_item = []
        for pk in (pks * 10)[:100]:
            t0 = time.time()
            get_similar_pks(pk, model, top_n=6)
            times_item.append((time.time() - t0) * 1000)

        return {
            'load_time_ms': round(load_time_ms, 1),
            'query_avg_ms': round(float(np.mean(times_query)), 3),
            'query_min_ms': round(float(np.min(times_query)), 3),
            'query_max_ms': round(float(np.max(times_query)), 3),
            'query_p95_ms': round(float(np.percentile(times_query, 95)), 3),
            'query_times': [round(t, 3) for t in times_query],
            'item_avg_ms': round(float(np.mean(times_item)), 3),
            'item_min_ms': round(float(np.min(times_item)), 3),
            'item_max_ms': round(float(np.max(times_item)), 3),
            'item_p95_ms': round(float(np.percentile(times_item, 95)), 3),
            'item_times': [round(t, 3) for t in times_item],
            'n_iterations': 100,
        }