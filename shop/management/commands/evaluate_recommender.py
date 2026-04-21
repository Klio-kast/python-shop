"""
Management command: evaluate_recommender
=========================================
Рассчитывает метрики качества рекомендательной модели TF-IDF + SVD:

  1. Hit Rate @ K      — доля запросов с релевантным ответом в топ-K
  2. MRR               — Mean Reciprocal Rank (позиция первого верного ответа)
  3. Coverage          — покрытие каталога (% товаров, попавших в рекомендации)
  4. Intra-list Similarity — разнообразие выдачи
  5. Reconstruction Error  — ошибка восстановления TF-IDF матрицы
  6. Note Drift        — совпадение нот запроса с нотами топ-1 ответа
  7. SVD n_components  — сравнение качества при разных размерностях

«Золотой стандарт» строится автоматически:
  Релевантным считается товар, у которого совпадает категория (семейство аромата)
  С запросом И пересечение нот с запросом не менее MIN_NOTE_OVERLAP слов.

Запуск:
    python manage.py evaluate_recommender

Результаты сохраняются в test_results/metrics_results.json
"""

import json
import time
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
from django.core.management.base import BaseCommand, CommandError


RESULTS_DIR  = Path('test_results')
RESULTS_FILE = RESULTS_DIR / 'metrics_results.json'
MODEL_PATH   = Path('ml_models/recommender_model.pkl')

# Минимальное пересечение нот для признания товара «релевантным»
MIN_NOTE_OVERLAP = 1

# Тестовые запросы с указанием ожидаемого семейства аромата
# Формат: (текст запроса, ожидаемое семейство, ключевые ноты)
TEST_QUERIES = [
    ('jasmine rose peony floral',          'floral',    ['jasmine', 'rose', 'peony']),
    ('woody cedar vetiver sandalwood',     'woody',     ['cedar', 'vetiver', 'sandalwood']),
    ('vanilla amber musk sweet gourmand',  'oriental',  ['vanilla', 'amber', 'musk']),
    ('citrus bergamot lemon fresh',        'citrus',    ['bergamot', 'lemon', 'citrus']),
    ('oud incense resin dark smoky',       'oud',       ['oud', 'incense', 'resin']),
    ('patchouli iris violet powder',       'chypre',    ['patchouli', 'iris', 'violet']),
    ('aquatic marine sea fresh clean',     'aromatic',  ['marine', 'sea', 'aquatic']),
    ('rose oud saffron spicy oriental',    'oriental',  ['rose', 'oud', 'saffron']),
    ('white musk skin cashmere soft',      'floral',    ['musk', 'cashmere']),
    ('bergamot neroli orange blossom',     'citrus',    ['bergamot', 'neroli', 'orange']),
]


class Command(BaseCommand):
    help = 'Рассчитывает метрики качества рекомендательной модели.'

    def handle(self, *args, **options):
        RESULTS_DIR.mkdir(exist_ok=True)

        self.stdout.write('\n' + '='*65)
        self.stdout.write('  ОЦЕНКА КАЧЕСТВА РЕКОМЕНДАТЕЛЬНОЙ МОДЕЛИ TF-IDF + SVD')
        self.stdout.write('='*65 + '\n')

        # Загрузка зависимостей
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            from shop.recommender import load_model, get_pks_by_query, get_similar_pks
            from shop.models import Product
        except ImportError as e:
            raise CommandError(f'Импорт: {e}')

        model = load_model()
        if model is None:
            raise CommandError(
                'Модель не обучена. Запустите: python manage.py train_recommender'
            )

        # Загрузка всех товаров
        all_products = list(
            Product.objects.select_related('category').order_by('id')
        )
        self.stdout.write(f'Товаров в каталоге: {len(all_products)}\n')

        results = {}

        # ── 1. Hit Rate @ K и MRR ────────────────────────────────
        self.stdout.write('► Метрика 1-2: Hit Rate@K и MRR ...')
        hr_mrr = self._calc_hit_rate_mrr(model, all_products, get_pks_by_query)
        results['hit_rate_mrr'] = hr_mrr
        for k in [1, 3, 5, 10]:
            self.stdout.write(
                f'  HR@{k:<2} = {hr_mrr["hit_rate"][str(k)]:.3f}  |  '
                f'MRR@{k} = {hr_mrr["mrr"][str(k)]:.3f}'
            )
        self.stdout.write()

        # ── 2. Coverage ──────────────────────────────────────────
        self.stdout.write('► Метрика 3: Coverage (покрытие каталога) ...')
        cov = self._calc_coverage(model, all_products, get_pks_by_query)
        results['coverage'] = cov
        self.stdout.write(
            f'  Coverage@10 = {cov["coverage_at_10"]:.1f}%  |  '
            f'Coverage@5  = {cov["coverage_at_5"]:.1f}%\n'
            f'  Уникальных товаров в рекомендациях: '
            f'{cov["unique_recommended"]} из {cov["total_products"]}'
        )
        self.stdout.write()

        # ── 3. Intra-list Similarity ─────────────────────────────
        self.stdout.write('► Метрика 4: Intra-list Similarity (разнообразие выдачи) ...')
        ils = self._calc_intra_list_similarity(model, all_products, get_pks_by_query)
        results['intra_list_similarity'] = ils
        self.stdout.write(
            f'  Средняя ILS = {ils["mean_ils"]:.4f}  '
            f'(идеал: 0.3–0.5, ближе к 0 = копии, к 1 = хаос)'
        )
        self.stdout.write()

        # ── 4. Reconstruction Error ──────────────────────────────
        self.stdout.write('► Метрика 5: Reconstruction Error TF-IDF матрицы ...')
        rec = self._calc_reconstruction_error(model)
        results['reconstruction_error'] = rec
        self.stdout.write(
            f'  Ошибка восстановления : {rec["reconstruction_error"]:.6f}\n'
            f'  Объяснённая дисперсия : {rec["explained_variance_ratio"]:.1f}%\n'
            f'  n_components          : {rec["n_components"]}'
        )
        self.stdout.write()

        # ── 5. Note Drift ────────────────────────────────────────
        self.stdout.write('► Метрика 6: Note Drift (дрейф нот) ...')
        nd = self._calc_note_drift(model, all_products, get_pks_by_query)
        results['note_drift'] = nd
        self.stdout.write(
            f'  Средний дрейф нот = {nd["mean_drift"]:.3f}  '
            f'(0 = идеально, 1 = полный дрейф)'
        )
        for q in nd['details']:
            self.stdout.write(
                f'  «{q["query"][:35]}» → '
                f'пересечение нот: {q["note_overlap"]:.0f}%'
            )
        self.stdout.write()

        # ── 6. SVD n_components сравнение ────────────────────────
        self.stdout.write('► Метрика 7: Влияние n_components на качество ...')
        svd_comp = self._calc_svd_comparison(model, all_products)
        results['svd_components'] = svd_comp
        for entry in svd_comp['comparison']:
            self.stdout.write(
                f'  n={entry["n_components"]:>4}: '
                f'HR@5={entry["hr_at_5"]:.3f}  '
                f'Coverage={entry["coverage"]:.1f}%  '
                f'ILS={entry["ils"]:.3f}'
            )
        self.stdout.write()

        # ── Сохранение ───────────────────────────────────────────
        results['meta'] = {
            'timestamp':    time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_products':   len(all_products),
            'n_queries':    len(TEST_QUERIES),
            'n_components': model.get('n_components', '?'),
            'vocab_size':   model.get('vocab_size', '?'),
        }

        with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        self.stdout.write('='*65)
        self.stdout.write(self.style.SUCCESS(
            f'\n✓ Результаты сохранены в {RESULTS_FILE}\n'
            f'  Для построения графиков запустите:\n'
            f'    python visualize_metrics.py\n'
        ))

    # ─────────────────────────────────────────────────────────────
    # Hit Rate @ K и MRR
    # ─────────────────────────────────────────────────────────────

    def _build_gold_standard(self, query_notes, query_family, all_products):
        """
        Для каждого тестового запроса определяет «релевантные» товары.
        Критерий: категория совпадает с ожидаемым семейством
                  ИЛИ ноты товара содержат хотя бы одно слово из запроса.
        """
        query_words = set(w.lower() for w in query_notes)
        relevant_pks = set()

        for p in all_products:
            cat = (p.category.name if p.category_id else '').lower()

            # Критерий 1: совпадение семейства
            family_match = any(fam in cat for fam in [query_family] +
                               [query_family[:4]])

            # Критерий 2: совпадение нот
            ing = str(getattr(p, 'ingredients', '') or '')
            frag = str(getattr(p, 'fragrances', '') or '')
            desc = str(getattr(p, 'description', '') or '')
            product_text = (ing + ' ' + frag + ' ' + desc).lower()
            note_overlap = sum(1 for w in query_words if w in product_text)

            if family_match or note_overlap >= MIN_NOTE_OVERLAP:
                relevant_pks.add(p.pk)

        return relevant_pks

    def _calc_hit_rate_mrr(self, model, all_products, get_pks_by_query):
        Ks = [1, 3, 5, 10]
        hit_rates = {str(k): [] for k in Ks}
        mrrs      = {str(k): [] for k in Ks}
        per_query = []

        for query_text, query_family, query_notes in TEST_QUERIES:
            relevant = self._build_gold_standard(query_notes, query_family, all_products)
            if not relevant:
                continue

            pairs = get_pks_by_query(query_text, model, top_n=max(Ks))
            ranked_pks = [pk for pk, _ in pairs]

            q_result = {
                'query': query_text[:40],
                'n_relevant': len(relevant),
                'hit_at': {},
                'rr_at': {},
            }

            for k in Ks:
                top_k = ranked_pks[:k]
                hit = 1 if any(pk in relevant for pk in top_k) else 0
                hit_rates[str(k)].append(hit)

                # Reciprocal Rank
                rr = 0.0
                for rank, pk in enumerate(top_k, start=1):
                    if pk in relevant:
                        rr = 1.0 / rank
                        break
                mrrs[str(k)].append(rr)
                q_result['hit_at'][str(k)] = hit
                q_result['rr_at'][str(k)] = round(rr, 4)

            per_query.append(q_result)

        return {
            'hit_rate': {k: round(float(np.mean(v)), 4) if v else 0
                         for k, v in hit_rates.items()},
            'mrr':      {k: round(float(np.mean(v)), 4) if v else 0
                         for k, v in mrrs.items()},
            'per_query': per_query,
            'n_queries': len(per_query),
        }

    # ─────────────────────────────────────────────────────────────
    # Coverage
    # ─────────────────────────────────────────────────────────────

    def _calc_coverage(self, model, all_products, get_pks_by_query):
        all_pks = {p.pk for p in all_products}
        seen_5  = set()
        seen_10 = set()

        # Прогоняем все тестовые запросы
        for query_text, _, _ in TEST_QUERIES:
            pairs = get_pks_by_query(query_text, model, top_n=10)
            ranked = [pk for pk, _ in pairs]
            seen_5.update(ranked[:5])
            seen_10.update(ranked[:10])

        # Дополнительно — item-to-item для каждого товара
        from shop.recommender import get_similar_pks
        for p in all_products[:50]:  # первые 50 как источники
            pairs = get_similar_pks(p.pk, model, top_n=6)
            seen_10.update(pk for pk, _ in pairs)

        total = len(all_pks)
        return {
            'total_products':    total,
            'unique_recommended': len(seen_10),
            'coverage_at_5':    round(len(seen_5)  / total * 100, 1),
            'coverage_at_10':   round(len(seen_10) / total * 100, 1),
            'not_recommended':  total - len(seen_10),
        }

    # ─────────────────────────────────────────────────────────────
    # Intra-list Similarity
    # ─────────────────────────────────────────────────────────────

    def _calc_intra_list_similarity(self, model, all_products, get_pks_by_query):
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim

        shop_norm = model['shop_reduced_norm']
        pks       = model['product_pks']
        pk_to_idx = {pk: i for i, pk in enumerate(pks)}

        ils_values = []
        per_query  = []

        for query_text, _, _ in TEST_QUERIES:
            pairs    = get_pks_by_query(query_text, model, top_n=10)
            top_pks  = [pk for pk, _ in pairs[:10]]
            indices  = [pk_to_idx[pk] for pk in top_pks if pk in pk_to_idx]

            if len(indices) < 2:
                continue

            vecs = shop_norm[indices]
            sim_matrix = cos_sim(vecs)
            # Берём верхний треугольник (без диагонали)
            n = len(indices)
            upper = [sim_matrix[i, j] for i in range(n) for j in range(i+1, n)]
            ils   = float(np.mean(upper)) if upper else 0.0
            ils_values.append(ils)
            per_query.append({'query': query_text[:40], 'ils': round(ils, 4)})

        mean_ils = float(np.mean(ils_values)) if ils_values else 0.0
        return {
            'mean_ils':  round(mean_ils, 4),
            'per_query': per_query,
            'ideal_min': 0.3,
            'ideal_max': 0.5,
        }

    # ─────────────────────────────────────────────────────────────
    # Reconstruction Error
    # ─────────────────────────────────────────────────────────────

    def _calc_reconstruction_error(self, model):
        vectorizer = model['vectorizer']
        svd        = model['svd']
        shop_norm  = model['shop_reduced_norm']

        # Берём тестовые строки — тексты запросов
        test_texts = [q[0] for q in TEST_QUERIES]
        X = vectorizer.transform(test_texts)

        # Проецируем в SVD-пространство и восстанавливаем обратно
        X_reduced   = svd.transform(X)
        X_reconstructed = X_reduced @ svd.components_

        # Frobenius norm ошибки
        diff = X.toarray() - X_reconstructed
        error = float(np.linalg.norm(diff, 'fro') / (X.shape[0] * X.shape[1]))

        # Объяснённая дисперсия
        explained = float(np.sum(svd.explained_variance_ratio_) * 100)

        return {
            'reconstruction_error':   round(error, 6),
            'explained_variance_ratio': round(explained, 2),
            'n_components':           model.get('n_components', len(svd.explained_variance_ratio_)),
            'singular_values':        [round(float(v), 4)
                                       for v in svd.singular_values_[:20]],
            'explained_per_component': [round(float(v) * 100, 3)
                                        for v in svd.explained_variance_ratio_[:20]],
        }

    # ─────────────────────────────────────────────────────────────
    # Note Drift
    # ─────────────────────────────────────────────────────────────

    def _calc_note_drift(self, model, all_products, get_pks_by_query):
        from shop.models import Product

        drift_values = []
        details      = []

        for query_text, _, query_notes in TEST_QUERIES:
            if not query_notes:
                continue

            query_words = set(w.lower() for w in query_notes)
            pairs       = get_pks_by_query(query_text, model, top_n=1)
            if not pairs:
                continue

            top_pk = pairs[0][0]
            try:
                p = Product.objects.get(pk=top_pk)
            except Exception:
                continue

            # Тексты нот топ-1 товара
            ing   = str(getattr(p, 'ingredients', '') or '').lower()
            frag  = str(getattr(p, 'fragrances',  '') or '').lower()
            desc  = str(getattr(p, 'description',  '') or '').lower()
            product_words = set((ing + ' ' + frag + ' ' + desc).split())

            # Пересечение
            overlap_count = sum(1 for w in query_words if w in product_words)
            overlap_pct   = (overlap_count / len(query_words) * 100) if query_words else 0
            drift         = 1 - overlap_pct / 100

            drift_values.append(drift)
            details.append({
                'query':        query_text[:40],
                'top1_product': p.name[:40],
                'query_notes':  list(query_words),
                'note_overlap': round(overlap_pct, 1),
                'drift':        round(drift, 4),
            })

        mean_drift = float(np.mean(drift_values)) if drift_values else 1.0
        return {
            'mean_drift': round(mean_drift, 4),
            'details':    details,
        }

    # ─────────────────────────────────────────────────────────────
    # SVD n_components сравнение
    # ─────────────────────────────────────────────────────────────

    def _calc_svd_comparison(self, model, all_products):
        """
        Сравниваем Hit Rate@5, Coverage и ILS при разных n_components,
        используя уже обученный vectorizer и масштабируя существующие SVD.
        """
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim

        vectorizer = model['vectorizer']
        pks        = model['product_pks']

        # Получаем признаковые строки для товаров магазина
        from shop.recommender import _shop_product_feature_string
        shop_strings = []
        valid_pks    = []
        for p in all_products:
            from shop.models import Product
            try:
                prod = Product.objects.select_related('category').get(pk=p.pk)
                fs = _shop_product_feature_string(prod)
                if fs.strip():
                    shop_strings.append(fs)
                    valid_pks.append(p.pk)
            except Exception:
                continue

        if not shop_strings:
            return {'comparison': [], 'error': 'Нет строк для сравнения'}

        shop_tfidf = vectorizer.transform(shop_strings)

        # Тестовые запросы
        test_texts   = [q[0] for q in TEST_QUERIES]
        query_family = [q[1] for q in TEST_QUERIES]
        query_notes  = [q[2] for q in TEST_QUERIES]

        # Строим золотой стандарт для HR@5
        from shop.models import Product as P
        all_prods_full = list(P.objects.select_related('category').filter(pk__in=valid_pks))

        comparison = []
        n_components_list = [10, 20, 50, model.get('n_components', 100)]
        # Убираем дубликаты и сортируем
        n_components_list = sorted(set(n_components_list))
        max_possible = min(shop_tfidf.shape[0] - 1, shop_tfidf.shape[1] - 1)

        for n_comp in n_components_list:
            if n_comp >= max_possible:
                n_comp = max_possible - 1
            if n_comp < 1:
                continue

            svd_tmp = TruncatedSVD(n_components=n_comp, random_state=42)
            # Обучаем на тех же данных, что и основная модель (на TF-IDF shop)
            shop_reduced = svd_tmp.fit_transform(shop_tfidf)
            shop_n = normalize(shop_reduced, norm='l2')
            shop_n = np.nan_to_num(shop_n, nan=0.0)

            pk_to_idx = {pk: i for i, pk in enumerate(valid_pks)}

            # HR@5
            hits = []
            seen = set()
            ils_vals = []

            for q_text, q_family, q_notes in TEST_QUERIES:
                q_tfidf   = vectorizer.transform([q_text])
                q_reduced = svd_tmp.transform(q_tfidf)
                from sklearn.preprocessing import normalize as norm_fn
                q_norm    = norm_fn(q_reduced, norm='l2')
                q_norm    = np.nan_to_num(q_norm, nan=0.0)

                scores    = (shop_n @ q_norm.T).flatten()
                top_idx   = np.argsort(scores)[::-1][:10]
                top_pks   = [valid_pks[i] for i in top_idx]

                seen.update(top_pks[:10])

                # Золотой стандарт
                relevant = self._build_gold_standard(q_notes, q_family, all_prods_full)
                hit = 1 if any(pk in relevant for pk in top_pks[:5]) else 0
                hits.append(hit)

                # ILS
                indices = [pk_to_idx[pk] for pk in top_pks[:10] if pk in pk_to_idx]
                if len(indices) >= 2:
                    vecs = shop_n[indices]
                    sim_m = cos_sim(vecs)
                    nn = len(indices)
                    upper = [sim_m[i, j] for i in range(nn) for j in range(i+1, nn)]
                    ils_vals.append(float(np.mean(upper)))

            coverage = len(seen) / len(valid_pks) * 100 if valid_pks else 0
            exp_var  = float(np.sum(svd_tmp.explained_variance_ratio_) * 100)

            comparison.append({
                'n_components':        n_comp,
                'hr_at_5':             round(float(np.mean(hits)), 4) if hits else 0,
                'coverage':            round(coverage, 1),
                'ils':                 round(float(np.mean(ils_vals)), 4) if ils_vals else 0,
                'explained_variance':  round(exp_var, 1),
            })

        return {'comparison': comparison}