"""
visualize_tests.py
==================
Читает результаты из test_results/recommender_test_results.json
и строит наглядные графики для дипломной работы.

Запуск (из корня проекта, рядом с manage.py):
    python visualize_tests.py

Требования:
    pip install matplotlib numpy
"""

import json
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

RESULTS_FILE  = Path('test_results/recommender_test_results.json')
OUTPUT_DIR    = Path('test_results')

# ── Цветовая палитра ─────────────────────────────────────────────
C_BLUE   = '#2563EB'
C_GREEN  = '#16A34A'
C_RED    = '#DC2626'
C_AMBER  = '#D97706'
C_GRAY   = '#6B7280'
C_LIGHT  = '#F3F4F6'
C_DARK   = '#1F2937'


def load_results() -> dict:
    if not RESULTS_FILE.exists():
        print(f'✗ Файл {RESULTS_FILE} не найден.')
        print('  Сначала запустите: python manage.py test_recommender')
        sys.exit(1)
    with open(RESULTS_FILE, encoding='utf-8') as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────
# График 1: Сводная таблица всех тестов
# ─────────────────────────────────────────────────────────────────

def plot_summary(data: dict):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')
    fig.patch.set_facecolor('white')

    summary = data.get('summary', {})
    t1 = data.get('test1_vectors', {})
    t2 = data.get('test2_relevance', {})
    t3 = data.get('test3_diversity', {})
    t4 = data.get('test4_item_to_item', {})
    t5 = data.get('test5_edge_cases', {})
    t6 = data.get('test6_performance', {})

    edge_ok = all(c['passed'] for c in t5.get('cases', []))
    perf_ok = t6.get('query_avg_ms', 999) < 50 and t6.get('item_avg_ms', 999) < 20

    rows = [
        ['Тест', 'Описание', 'Результат', 'Статус'],
        ['1. Векторы',
         f'{t1.get("nonzero", 0)} из {t1.get("total", 0)} товаров (норма: {t1.get("mean_norm", 0):.3f})',
         f'{t1.get("nonzero", 0)}/{t1.get("total", 0)}',
         '✓ Пройден' if t1.get('passed') else '✗ Провален'],
        ['2. Релевантность',
         f'{t2.get("passed_count", 0)} из 5 запросов дали релевантные результаты',
         f'{t2.get("passed_count", 0)}/5',
         '✓ Пройден' if t2.get('overall_passed') else '✗ Провален'],
        ['3. Различие',
         'Запросы по разным нотам возвращают разные товары',
         f'{sum(p["passed"] for p in t3.get("pairs", []))}/3',
         '✓ Пройден' if t3.get('passed') else '✗ Провален'],
        ['4. Item-to-item',
         f'{t4.get("passed_count", 0)} из 5 товаров получили похожие рекомендации',
         f'{t4.get("passed_count", 0)}/5',
         '✓ Пройден' if t4.get('overall_passed') else '✗ Провален'],
        ['5. Граничные случаи',
         f'{sum(c["passed"] for c in t5.get("cases", []))} из 5 случаев без ошибок',
         f'{sum(c["passed"] for c in t5.get("cases", []))}/5',
         '✓ Пройден' if edge_ok else '~ Частично'],
        ['6. Производительность',
         f'Запрос: {t6.get("query_avg_ms", 0):.2f} мс, Item: {t6.get("item_avg_ms", 0):.2f} мс',
         f'{t6.get("query_avg_ms", 0):.2f} мс',
         '✓ Пройден' if perf_ok else '~ Приемлемо'],
    ]

    col_widths = [0.18, 0.50, 0.16, 0.16]
    col_x = [0.01]
    for w in col_widths[:-1]:
        col_x.append(col_x[-1] + w)

    # Заголовок
    fig.text(0.5, 0.95, 'Сводные результаты тестирования рекомендательной модели',
             ha='center', va='top', fontsize=14, fontweight='bold', color=C_DARK)
    fig.text(0.5, 0.89, f'Дата: {summary.get("timestamp", "")}  |  '
             f'Товаров: {summary.get("n_products", "")}  |  '
             f'Словарь TF-IDF: {summary.get("vocab_size", "")} токенов  |  '
             f'SVD компоненты: {summary.get("n_components", "")}',
             ha='center', va='top', fontsize=9, color=C_GRAY)

    row_height = 0.10
    row_y_start = 0.82

    for i, row in enumerate(rows):
        y = row_y_start - i * row_height
        is_header = (i == 0)

        # Фон строки
        bg_color = C_DARK if is_header else (C_LIGHT if i % 2 == 0 else 'white')
        rect = mpatches.FancyBboxPatch(
            (0, y - row_height * 0.9), 1, row_height * 0.92,
            boxstyle='round,pad=0.01', linewidth=0,
            facecolor=bg_color, transform=ax.transAxes, clip_on=False
        )
        ax.add_patch(rect)

        for j, (cell, x) in enumerate(zip(row, col_x)):
            color = 'white' if is_header else C_DARK
            weight = 'bold' if is_header else 'normal'
            align = 'left'

            if not is_header and j == 3:
                color = C_GREEN if '✓' in cell else (C_RED if '✗' in cell else C_AMBER)
                weight = 'bold'

            ax.text(x + 0.005, y - row_height * 0.35, cell,
                    transform=ax.transAxes,
                    fontsize=8.5 if j == 1 else 9,
                    color=color, fontweight=weight,
                    va='center', ha=align)

    # Итог
    passed = summary.get('passed', 0)
    total  = summary.get('total', 6)
    y_foot = row_y_start - len(rows) * row_height + 0.02
    fig.text(0.5, y_foot,
             f'Итого пройдено: {passed} из {total} тестов',
             ha='center', fontsize=11, fontweight='bold',
             color=C_GREEN if passed >= 5 else C_AMBER)

    plt.tight_layout()
    out = OUTPUT_DIR / 'plot1_summary_table.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  ✓ {out}')


# ─────────────────────────────────────────────────────────────────
# График 2: Распределение норм векторов (гистограмма)
# ─────────────────────────────────────────────────────────────────

def plot_vector_norms(data: dict):
    t1 = data.get('test1_vectors', {})
    dist = t1.get('norm_distribution', {})
    if not dist:
        return

    labels = list(dist.keys())
    values = list(dist.values())
    total  = sum(values)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('white')
    fig.suptitle('Тест 1: Распределение норм векторов товаров магазина',
                 fontsize=13, fontweight='bold', color=C_DARK, y=1.02)

    # Левый: столбчатая диаграмма
    ax = axes[0]
    bars = ax.bar(labels, values, color=C_BLUE, edgecolor='white', linewidth=0.5, width=0.6)
    ax.set_xlabel('Диапазон нормы вектора', fontsize=10)
    ax.set_ylabel('Количество товаров', fontsize=10)
    ax.set_title('Количество товаров по диапазонам норм', fontsize=11)
    ax.set_facecolor(C_LIGHT)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='x', rotation=30)

    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(val), ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Правый: сводная статистика в виде плиток
    ax2 = axes[1]
    ax2.axis('off')
    ax2.set_title('Статистика векторов', fontsize=11)

    stats = [
        ('Всего товаров',        str(t1.get('total', 0)),    C_BLUE),
        ('Ненулевых векторов',   str(t1.get('nonzero', 0)),  C_GREEN),
        ('Нулевых векторов',     str(t1.get('zero', 0)),     C_RED if t1.get('zero', 0) > 0 else C_GRAY),
        ('Средняя норма',        f'{t1.get("mean_norm", 0):.4f}', C_BLUE),
        ('Минимальная норма',    f'{t1.get("min_norm", 0):.4f}', C_GRAY),
        ('Максимальная норма',   f'{t1.get("max_norm", 0):.4f}', C_GRAY),
        ('% ненулевых',
         f'{t1.get("nonzero", 0) / t1.get("total", 1) * 100:.1f}%',
         C_GREEN if t1.get('passed') else C_RED),
    ]

    for i, (label, value, color) in enumerate(stats):
        y = 0.85 - i * 0.12
        rect = mpatches.FancyBboxPatch((0.05, y - 0.04), 0.9, 0.10,
                                        boxstyle='round,pad=0.01',
                                        facecolor=C_LIGHT, edgecolor=color, linewidth=1.5,
                                        transform=ax2.transAxes, clip_on=False)
        ax2.add_patch(rect)
        ax2.text(0.15, y + 0.01, label, transform=ax2.transAxes,
                 fontsize=9.5, color=C_DARK, va='center')
        ax2.text(0.85, y + 0.01, value, transform=ax2.transAxes,
                 fontsize=10, fontweight='bold', color=color, va='center', ha='right')

    plt.tight_layout()
    out = OUTPUT_DIR / 'plot2_vector_norms.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  ✓ {out}')


# ─────────────────────────────────────────────────────────────────
# График 3: Релевантность запросов (горизонтальные бары)
# ─────────────────────────────────────────────────────────────────

def plot_relevance(data: dict):
    t2 = data.get('test2_relevance', {})
    queries = t2.get('queries', [])
    if not queries:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')
    fig.suptitle('Тест 2: Релевантность результатов поиска по нотам',
                 fontsize=13, fontweight='bold', color=C_DARK, y=1.01)

    labels        = [f'«{q["query"][:30]}»' for q in queries]
    match_pcts    = [q['category_match_pct'] for q in queries]
    top_scores    = [q['top_score'] for q in queries]

    # Левый: совпадение категорий
    ax1 = axes[0]
    colors = [C_GREEN if p >= 50 else (C_AMBER if p >= 30 else C_RED) for p in match_pcts]
    bars = ax1.barh(labels, match_pcts, color=colors, edgecolor='white', height=0.5)
    ax1.axvline(50, color=C_GREEN, linestyle='--', alpha=0.5, linewidth=1, label='Порог 50%')
    ax1.axvline(30, color=C_AMBER, linestyle='--', alpha=0.5, linewidth=1, label='Порог 30%')
    ax1.set_xlabel('Совпадение категорий, %', fontsize=10)
    ax1.set_title('Совпадение категорий результатов\nс ожидаемым семейством аромата', fontsize=10)
    ax1.set_xlim(0, 105)
    ax1.set_facecolor(C_LIGHT)
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.legend(fontsize=8)
    for bar, val in zip(bars, match_pcts):
        ax1.text(val + 1, bar.get_y() + bar.get_height()/2,
                 f'{val:.0f}%', va='center', fontsize=9, fontweight='bold')

    # Правый: максимальная схожесть топ-1 результата
    ax2 = axes[1]
    colors2 = [C_BLUE if s >= 10 else (C_AMBER if s >= 5 else C_RED) for s in top_scores]
    bars2 = ax2.barh(labels, top_scores, color=colors2, edgecolor='white', height=0.5)
    ax2.axvline(10, color=C_BLUE, linestyle='--', alpha=0.5, linewidth=1, label='Порог 10%')
    ax2.set_xlabel('Схожесть топ-1 результата, %', fontsize=10)
    ax2.set_title('Максимальная схожесть\nпервого результата с запросом', fontsize=10)
    ax2.set_xlim(0, max(top_scores) * 1.25 + 5 if top_scores else 50)
    ax2.set_facecolor(C_LIGHT)
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.legend(fontsize=8)
    for bar, val in zip(bars2, top_scores):
        ax2.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                 f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    out = OUTPUT_DIR / 'plot3_relevance.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  ✓ {out}')


# ─────────────────────────────────────────────────────────────────
# График 4: Различие результатов (тепловая матрица)
# ─────────────────────────────────────────────────────────────────

def plot_diversity(data: dict):
    t3 = data.get('test3_diversity', {})
    pairs = t3.get('pairs', [])
    if not pairs:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('white')
    fig.suptitle('Тест 3: Различие результатов для разных запросов',
                 fontsize=13, fontweight='bold', color=C_DARK)

    # Левый: % уникальных результатов
    ax1 = axes[0]
    pair_labels = [f'Пара {i+1}' for i in range(len(pairs))]
    unique_pcts = [p['unique_pct'] for p in pairs]
    overlaps    = [p['overlap'] for p in pairs]

    x = np.arange(len(pairs))
    width = 0.35
    b1 = ax1.bar(x - width/2, unique_pcts, width, label='Уникальных, %',
                  color=C_GREEN, edgecolor='white')
    b2 = ax1.bar(x + width/2, [o/5*100 for o in overlaps], width,
                  label='Совпадений, %', color=C_RED, edgecolor='white')
    ax1.set_xticks(x)
    ax1.set_xticklabels(pair_labels)
    ax1.set_ylabel('Процент от 5 результатов')
    ax1.set_title('Уникальность результатов\nдля попарно разных запросов')
    ax1.set_ylim(0, 115)
    ax1.set_facecolor(C_LIGHT)
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.legend()
    ax1.axhline(80, color=C_BLUE, linestyle='--', alpha=0.4, linewidth=1)
    ax1.text(len(pairs)-0.5, 82, 'Цель: 80%+', fontsize=8, color=C_BLUE)

    for bar, val in zip(b1, unique_pcts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.0f}%', ha='center', fontsize=9, fontweight='bold')

    # Правый: описание пар
    ax2 = axes[1]
    ax2.axis('off')
    ax2.set_title('Тестируемые пары запросов', fontsize=11)

    for i, pair in enumerate(pairs):
        y = 0.85 - i * 0.28
        color = C_GREEN if pair['passed'] else C_RED
        mark  = '✓' if pair['passed'] else '✗'

        rect = mpatches.FancyBboxPatch((0.02, y - 0.14), 0.96, 0.26,
                                        boxstyle='round,pad=0.01',
                                        facecolor=C_LIGHT, edgecolor=color, linewidth=1.5,
                                        transform=ax2.transAxes, clip_on=False)
        ax2.add_patch(rect)

        ax2.text(0.05, y + 0.07, f'{mark} Пара {i+1}', transform=ax2.transAxes,
                 fontsize=9.5, fontweight='bold', color=color, va='center')
        ax2.text(0.05, y - 0.01, f'«{pair["q1"][:35]}»', transform=ax2.transAxes,
                 fontsize=8, color=C_DARK, va='center')
        ax2.text(0.05, y - 0.08, f'«{pair["q2"][:35]}»', transform=ax2.transAxes,
                 fontsize=8, color=C_GRAY, va='center')
        ax2.text(0.96, y - 0.01,
                 f'Совпадений: {pair["overlap"]}/5\nУникальных: {pair["unique_pct"]:.0f}%',
                 transform=ax2.transAxes, fontsize=8, color=C_DARK,
                 va='center', ha='right')

    plt.tight_layout()
    out = OUTPUT_DIR / 'plot4_diversity.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  ✓ {out}')


# ─────────────────────────────────────────────────────────────────
# График 5: Граничные случаи
# ─────────────────────────────────────────────────────────────────

def plot_edge_cases(data: dict):
    t5 = data.get('test5_edge_cases', {})
    cases = t5.get('cases', [])
    if not cases:
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor('white')
    ax.axis('off')
    ax.set_title('Тест 5: Устойчивость к граничным случаям',
                 fontsize=13, fontweight='bold', color=C_DARK, pad=15)

    cols = ['Тест', 'Входные данные', 'Результат', 'Статус']
    col_x = [0.01, 0.22, 0.70, 0.88]
    col_w = [0.21, 0.48, 0.18, 0.12]

    # Заголовок таблицы
    for txt, x in zip(cols, col_x):
        rect = mpatches.FancyBboxPatch((x, 0.88), col_w[cols.index(txt)], 0.09,
                                        boxstyle='round,pad=0.005',
                                        facecolor=C_DARK, linewidth=0,
                                        transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        ax.text(x + 0.01, 0.925, txt, transform=ax.transAxes,
                fontsize=9.5, fontweight='bold', color='white', va='center')

    for i, case in enumerate(cases):
        y = 0.78 - i * 0.14
        bg = C_LIGHT if i % 2 == 0 else 'white'
        border = C_GREEN if case['passed'] else C_RED

        rect = mpatches.FancyBboxPatch((0.01, y - 0.04), 0.98, 0.12,
                                        boxstyle='round,pad=0.005',
                                        facecolor=bg, edgecolor=border, linewidth=1,
                                        transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)

        ax.text(col_x[0] + 0.01, y + 0.02, case['name'],
                transform=ax.transAxes, fontsize=8.5, color=C_DARK, va='center')
        ax.text(col_x[1] + 0.01, y + 0.02, case['description'][:55],
                transform=ax.transAxes, fontsize=8, color=C_GRAY, va='center')
        ax.text(col_x[2] + 0.01, y + 0.02,
                f'n={case["n_results"]}, max={case["max_score"]:.1f}%',
                transform=ax.transAxes, fontsize=8, color=C_DARK, va='center')

        status_text = '✓ Пройден' if case['passed'] else '✗ Ошибка'
        ax.text(col_x[3] + 0.01, y + 0.02, status_text,
                transform=ax.transAxes, fontsize=8.5, fontweight='bold',
                color=C_GREEN if case['passed'] else C_RED, va='center')

    # Итог
    passed = sum(c['passed'] for c in cases)
    ax.text(0.5, 0.02, f'Пройдено: {passed} из {len(cases)} тестов',
            transform=ax.transAxes, fontsize=10, fontweight='bold',
            color=C_GREEN if passed == len(cases) else C_AMBER,
            va='center', ha='center')

    plt.tight_layout()
    out = OUTPUT_DIR / 'plot5_edge_cases.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  ✓ {out}')


# ─────────────────────────────────────────────────────────────────
# График 6: Производительность
# ─────────────────────────────────────────────────────────────────

def plot_performance(data: dict):
    t6 = data.get('test6_performance', {})
    query_times = t6.get('query_times', [])
    item_times  = t6.get('item_times', [])
    if not query_times:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('white')
    fig.suptitle('Тест 6: Производительность рекомендательной модели (100 итераций)',
                 fontsize=13, fontweight='bold', color=C_DARK)

    # Левый: гистограмма времени поиска по запросу
    ax1 = axes[0]
    ax1.hist(query_times, bins=20, color=C_BLUE, edgecolor='white', alpha=0.85)
    ax1.axvline(np.mean(query_times), color=C_RED, linestyle='--',
                linewidth=1.5, label=f'Среднее: {np.mean(query_times):.2f} мс')
    ax1.axvline(np.percentile(query_times, 95), color=C_AMBER, linestyle=':',
                linewidth=1.5, label=f'95-й перц.: {np.percentile(query_times, 95):.2f} мс')
    ax1.set_xlabel('Время, мс')
    ax1.set_ylabel('Количество запросов')
    ax1.set_title('Поиск по тексту запроса\n(get_pks_by_query)')
    ax1.set_facecolor(C_LIGHT)
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.legend(fontsize=8)

    # Средний: гистограмма item-to-item
    ax2 = axes[1]
    ax2.hist(item_times, bins=20, color=C_GREEN, edgecolor='white', alpha=0.85)
    ax2.axvline(np.mean(item_times), color=C_RED, linestyle='--',
                linewidth=1.5, label=f'Среднее: {np.mean(item_times):.2f} мс')
    ax2.axvline(np.percentile(item_times, 95), color=C_AMBER, linestyle=':',
                linewidth=1.5, label=f'95-й перц.: {np.percentile(item_times, 95):.2f} мс')
    ax2.set_xlabel('Время, мс')
    ax2.set_ylabel('Количество запросов')
    ax2.set_title('Item-to-item поиск\n(get_similar_pks)')
    ax2.set_facecolor(C_LIGHT)
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.legend(fontsize=8)

    # Правый: сводная таблица метрик
    ax3 = axes[2]
    ax3.axis('off')
    ax3.set_title('Сводные метрики производительности', fontsize=10)

    metrics = [
        ('Загрузка модели',      f'{t6.get("load_time_ms", 0):.1f} мс',    C_BLUE),
        ('',                     '',                                          'white'),
        ('Поиск по запросу:',    '',                                          C_DARK),
        ('  Среднее время',      f'{t6.get("query_avg_ms", 0):.3f} мс',     C_BLUE),
        ('  Минимум',            f'{t6.get("query_min_ms", 0):.3f} мс',     C_GRAY),
        ('  Максимум',           f'{t6.get("query_max_ms", 0):.3f} мс',     C_GRAY),
        ('  95-й перцентиль',    f'{t6.get("query_p95_ms", 0):.3f} мс',     C_AMBER),
        ('',                     '',                                          'white'),
        ('Item-to-item:',        '',                                          C_DARK),
        ('  Среднее время',      f'{t6.get("item_avg_ms", 0):.3f} мс',      C_GREEN),
        ('  Минимум',            f'{t6.get("item_min_ms", 0):.3f} мс',      C_GRAY),
        ('  Максимум',           f'{t6.get("item_max_ms", 0):.3f} мс',      C_GRAY),
        ('  95-й перцентиль',    f'{t6.get("item_p95_ms", 0):.3f} мс',      C_AMBER),
        ('',                     '',                                          'white'),
        ('Итераций в тесте',     str(t6.get('n_iterations', 100)),           C_DARK),
    ]

    for i, (label, value, color) in enumerate(metrics):
        y = 0.97 - i * 0.063
        if label and value:
            rect = mpatches.FancyBboxPatch((0.02, y - 0.03), 0.96, 0.055,
                                            boxstyle='round,pad=0.005',
                                            facecolor=C_LIGHT if i % 2 == 0 else 'white',
                                            linewidth=0,
                                            transform=ax3.transAxes, clip_on=False)
            ax3.add_patch(rect)
        ax3.text(0.05, y, label, transform=ax3.transAxes,
                 fontsize=8.5, color=color, va='center')
        ax3.text(0.95, y, value, transform=ax3.transAxes,
                 fontsize=9, fontweight='bold', color=color, va='center', ha='right')

    plt.tight_layout()
    out = OUTPUT_DIR / 'plot6_performance.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  ✓ {out}')


# ─────────────────────────────────────────────────────────────────
# График 7: Item-to-item схожесть по товарам
# ─────────────────────────────────────────────────────────────────

def plot_item_to_item(data: dict):
    t4 = data.get('test4_item_to_item', {})
    items = t4.get('items', [])
    if not items:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')
    fig.suptitle('Тест 4: Item-to-item рекомендации (схожесть похожих товаров)',
                 fontsize=13, fontweight='bold', color=C_DARK)

    # Левый: средняя схожесть по каждому товару
    ax1 = axes[0]
    names      = [it['product_name'][:25] + '…' if len(it['product_name']) > 25
                  else it['product_name'] for it in items]
    mean_scores = [it['mean_score'] for it in items]
    cat_matches = [it['category_match_pct'] for it in items]

    x = np.arange(len(items))
    w = 0.35
    b1 = ax1.bar(x - w/2, mean_scores, w, label='Средняя схожесть, %',
                  color=C_BLUE, edgecolor='white')
    b2 = ax1.bar(x + w/2, cat_matches, w, label='Совпадение категорий, %',
                  color=C_GREEN, edgecolor='white')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20, ha='right', fontsize=8)
    ax1.set_ylabel('Процент, %')
    ax1.set_title('Средняя схожесть и совпадение\nкатегорий для 5 тестовых товаров')
    ax1.set_facecolor(C_LIGHT)
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.legend(fontsize=9)
    for bar, val in zip(b1, mean_scores):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val:.1f}', ha='center', fontsize=8)

    # Правый: детальные оценки для каждого товара (линии убывания)
    ax2 = axes[1]
    colors_list = [C_BLUE, C_GREEN, C_RED, C_AMBER, C_GRAY]

    for i, item in enumerate(items):
        scores = item.get('scores', [])
        if scores:
            ax2.plot(range(1, len(scores)+1), scores,
                     'o-', color=colors_list[i % len(colors_list)],
                     linewidth=1.8, markersize=6,
                     label=f'{item["product_name"][:20]}…' if len(item["product_name"]) > 20
                     else item["product_name"])

    ax2.set_xlabel('Позиция рекомендации')
    ax2.set_ylabel('Схожесть, %')
    ax2.set_title('Убывание схожести\nпо позициям рекомендаций')
    ax2.set_xticks(range(1, 7))
    ax2.set_xticklabels([f'#{i}' for i in range(1, 7)])
    ax2.set_facecolor(C_LIGHT)
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.legend(fontsize=7.5, loc='upper right')
    ax2.axhline(0, color=C_GRAY, linestyle='-', alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    out = OUTPUT_DIR / 'plot7_item_to_item.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  ✓ {out}')


# ─────────────────────────────────────────────────────────────────
# Главная функция
# ─────────────────────────────────────────────────────────────────

def main():
    print('\n' + '='*55)
    print('  ПОСТРОЕНИЕ ГРАФИКОВ ПО РЕЗУЛЬТАТАМ ТЕСТИРОВАНИЯ')
    print('='*55)

    data = load_results()
    summary = data.get('summary', {})
    print(f'\nДата тестирования : {summary.get("timestamp", "—")}')
    print(f'Товаров в модели  : {summary.get("n_products", "—")}')
    print(f'Пройдено тестов   : {summary.get("passed", "—")}/{summary.get("total", "—")}')
    print(f'\nСохраняю графики в папку {OUTPUT_DIR}/ ...\n')

    plot_summary(data)
    plot_vector_norms(data)
    plot_relevance(data)
    plot_diversity(data)
    plot_edge_cases(data)
    plot_performance(data)
    plot_item_to_item(data)

    print(f'\n✓ Все графики сохранены в папку {OUTPUT_DIR}/')
    print('\nФайлы:')
    for f in sorted(OUTPUT_DIR.glob('plot*.png')):
        size_kb = f.stat().st_size // 1024
        print(f'  {f.name:45s}  {size_kb} KB')

    print('\nГотово! Используйте файлы PNG для презентации.')
    print('='*55 + '\n')


if __name__ == '__main__':
    main()