"""
visualize_metrics.py
=====================
Строит графики по результатам evaluate_recommender.
Запуск из корня проекта:
    python visualize_metrics.py

Требования: pip install matplotlib numpy
"""

import json
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

RESULTS_FILE = Path('test_results/metrics_results.json')
OUTPUT_DIR   = Path('test_results')

C_BLUE  = '#2563EB'
C_GREEN = '#16A34A'
C_RED   = '#DC2626'
C_AMBER = '#D97706'
C_GRAY  = '#9CA3AF'
C_LIGHT = '#F3F4F6'
C_DARK  = '#111827'
C_PURPLE= '#7C3AED'


def load():
    if not RESULTS_FILE.exists():
        print(f'Файл {RESULTS_FILE} не найден.')
        print('Запустите: python manage.py evaluate_recommender')
        sys.exit(1)
    with open(RESULTS_FILE, encoding='utf-8') as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────
# ГРАФИК 1: Hit Rate @ K и MRR — главная сводка
# ─────────────────────────────────────────────────────────────────

def plot_hit_rate_mrr(data):
    hr_mrr = data.get('hit_rate_mrr', {})
    Ks = [1, 3, 5, 10]

    hr_vals  = [hr_mrr.get('hit_rate', {}).get(str(k), 0) for k in Ks]
    mrr_vals = [hr_mrr.get('mrr', {}).get(str(k), 0) for k in Ks]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')
    fig.suptitle('Метрики Hit Rate@K и MRR@K\n'
                 'Оценка полноты и качества ранжирования рекомендаций',
                 fontsize=13, fontweight='bold', color=C_DARK, y=1.01)

    # ── HR@K ──────────────────────────────────────────────────
    ax1 = axes[0]
    bars = ax1.bar([f'HR@{k}' for k in Ks], hr_vals,
                   color=[C_GREEN if v >= 0.7 else C_AMBER if v >= 0.4 else C_RED
                          for v in hr_vals],
                   edgecolor='white', width=0.5)
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel('Hit Rate', fontsize=11)
    ax1.set_title('Hit Rate @ K\n'
                  '(доля запросов с релевантным\nответом в топ-K)',
                  fontsize=10)
    ax1.axhline(0.7, color=C_GREEN, linestyle='--', alpha=0.5, linewidth=1.2,
                label='Хорошо ≥ 0.7')
    ax1.axhline(0.4, color=C_AMBER, linestyle='--', alpha=0.5, linewidth=1.2,
                label='Приемлемо ≥ 0.4')
    ax1.set_facecolor(C_LIGHT)
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.legend(fontsize=9)
    for bar, val in zip(bars, hr_vals):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.025,
                 f'{val:.3f}', ha='center', fontsize=11, fontweight='bold',
                 color=C_DARK)

    # ── MRR@K ────────────────────────────────────────────────
    ax2 = axes[1]
    bars2 = ax2.bar([f'MRR@{k}' for k in Ks], mrr_vals,
                    color=C_BLUE, edgecolor='white', width=0.5, alpha=0.85)
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel('MRR', fontsize=11)
    ax2.set_title('Mean Reciprocal Rank @ K\n'
                  '(позиция первого релевантного\nответа в топ-K)',
                  fontsize=10)
    ax2.axhline(0.5, color=C_BLUE, linestyle='--', alpha=0.4, linewidth=1.2,
                label='Ориентир ≥ 0.5')
    ax2.set_facecolor(C_LIGHT)
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.legend(fontsize=9)
    for bar, val in zip(bars2, mrr_vals):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.025,
                 f'{val:.3f}', ha='center', fontsize=11, fontweight='bold',
                 color=C_DARK)

    # Пояснение под графиком
    fig.text(0.5, -0.04,
             'HR@K — вопрос «нашли ли мы хоть что-то полезное?»  |  '
             'MRR@K — вопрос «насколько высоко стоит лучший ответ?»\n'
             f'Тестовых запросов: {hr_mrr.get("n_queries", "?")}',
             ha='center', fontsize=9, color=C_GRAY)

    plt.tight_layout()
    out = OUTPUT_DIR / 'metric1_hit_rate_mrr.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  ✓ {out}')


# ─────────────────────────────────────────────────────────────────
# ГРАФИК 2: Hit Rate по каждому запросу (тепловая карта)
# ─────────────────────────────────────────────────────────────────

def plot_per_query_heatmap(data):
    hr_mrr    = data.get('hit_rate_mrr', {})
    per_query = hr_mrr.get('per_query', [])
    if not per_query:
        return

    Ks = ['1', '3', '5', '10']
    queries = [q['query'][:35] for q in per_query]
    matrix  = np.array([[q['hit_at'].get(k, 0) for k in Ks] for q in per_query],
                       dtype=float)

    fig, ax = plt.subplots(figsize=(10, max(5, len(queries) * 0.7)))
    fig.patch.set_facecolor('white')

    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(Ks)))
    ax.set_xticklabels([f'HR@{k}' for k in Ks], fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(queries)))
    ax.set_yticklabels(queries, fontsize=9)
    ax.set_title('Hit Rate по каждому запросу\n'
                 '(зелёный = нашли релевантный аромат, красный = не нашли)',
                 fontsize=12, fontweight='bold', color=C_DARK, pad=12)

    for i in range(len(queries)):
        for j in range(len(Ks)):
            val = matrix[i, j]
            ax.text(j, i, '✓' if val == 1 else '✗',
                    ha='center', va='center', fontsize=14,
                    color='white' if val == 1 else C_DARK, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Hit (1 = найдено, 0 = не найдено)',
                 fraction=0.03, pad=0.02)
    plt.tight_layout()
    out = OUTPUT_DIR / 'metric2_per_query_heatmap.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  ✓ {out}')


# ─────────────────────────────────────────────────────────────────
# ГРАФИК 3: Coverage
# ─────────────────────────────────────────────────────────────────

def plot_coverage(data):
    cov = data.get('coverage', {})
    total   = cov.get('total_products', 1)
    seen_5  = round(cov.get('coverage_at_5',  0) / 100 * total)
    seen_10 = round(cov.get('coverage_at_10', 0) / 100 * total)
    not_rec = cov.get('not_recommended', total)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('white')
    fig.suptitle('Метрика Coverage — покрытие каталога рекомендациями',
                 fontsize=13, fontweight='bold', color=C_DARK)

    # ── Pie chart ──────────────────────────────────────────────
    ax1 = axes[0]
    sizes  = [seen_10, not_rec]
    labels = [f'Рекомендовано\n({seen_10} товаров)', f'Не рекомендовано\n({not_rec} товаров)']
    colors_pie = [C_GREEN, C_LIGHT]
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, colors=colors_pie,
        autopct='%1.1f%%', startangle=90,
        wedgeprops=dict(edgecolor='white', linewidth=2),
        textprops=dict(fontsize=10)
    )
    autotexts[0].set_fontweight('bold')
    autotexts[0].set_fontsize(13)
    autotexts[0].set_color(C_DARK)
    ax1.set_title(f'Coverage@10\n(все {total} товаров каталога)',
                  fontsize=11, fontweight='bold')

    # ── Bar comparison ─────────────────────────────────────────
    ax2 = axes[1]
    labels_bar = ['Coverage@5', 'Coverage@10']
    values_bar = [cov.get('coverage_at_5', 0), cov.get('coverage_at_10', 0)]
    colors_bar = [C_AMBER if v < 30 else C_GREEN for v in values_bar]

    bars = ax2.bar(labels_bar, values_bar, color=colors_bar,
                   edgecolor='white', width=0.4)
    ax2.axhline(30, color=C_RED, linestyle='--', alpha=0.6, linewidth=1.2,
                label='Критический порог: 30%')
    ax2.axhline(50, color=C_GREEN, linestyle='--', alpha=0.6, linewidth=1.2,
                label='Хорошее покрытие: 50%')
    ax2.set_ylim(0, 115)
    ax2.set_ylabel('Покрытие каталога, %', fontsize=11)
    ax2.set_title('Доля товаров, попавших\nв рекомендации', fontsize=11)
    ax2.set_facecolor(C_LIGHT)
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.legend(fontsize=9)
    for bar, val in zip(bars, values_bar):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 2,
                 f'{val:.1f}%', ha='center', fontsize=13, fontweight='bold')

    fig.text(0.5, -0.03,
             'Coverage < 30%: модель рекомендует одни и те же "популярные" ароматы на любой запрос\n'
             'Coverage > 50%: модель хорошо покрывает каталог',
             ha='center', fontsize=9, color=C_GRAY)

    plt.tight_layout()
    out = OUTPUT_DIR / 'metric3_coverage.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  ✓ {out}')


# ─────────────────────────────────────────────────────────────────
# ГРАФИК 4: Intra-list Similarity
# ─────────────────────────────────────────────────────────────────

def plot_ils(data):
    ils_data   = data.get('intra_list_similarity', {})
    per_query  = ils_data.get('per_query', [])
    mean_ils   = ils_data.get('mean_ils', 0)

    if not per_query:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('white')
    fig.suptitle('Метрика Intra-list Similarity — разнообразие выдачи',
                 fontsize=13, fontweight='bold', color=C_DARK)

    queries = [q['query'][:30] for q in per_query]
    ils_vals = [q['ils'] for q in per_query]

    # ── Горизонтальные бары ────────────────────────────────────
    ax1 = axes[0]
    colors_bar = []
    for v in ils_vals:
        if v < 0.2:
            colors_bar.append(C_RED)    # слишком похожи — дубли
        elif v <= 0.6:
            colors_bar.append(C_GREEN)  # идеальный диапазон
        else:
            colors_bar.append(C_AMBER)  # слишком разные

    bars = ax1.barh(queries, ils_vals, color=colors_bar, edgecolor='white', height=0.6)
    ax1.axvline(0.3, color=C_GREEN, linestyle='--', alpha=0.6, linewidth=1.5,
                label='Нижний порог: 0.3')
    ax1.axvline(0.5, color=C_GREEN, linestyle='--', alpha=0.6, linewidth=1.5,
                label='Верхний порог: 0.5')
    ax1.axvline(mean_ils, color=C_BLUE, linestyle='-', alpha=0.8, linewidth=2,
                label=f'Среднее: {mean_ils:.4f}')
    ax1.set_xlabel('Intra-list Similarity (косинусное сходство внутри выдачи)', fontsize=10)
    ax1.set_title('ILS по каждому запросу', fontsize=11)
    ax1.set_xlim(0, 1.05)
    ax1.set_facecolor(C_LIGHT)
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.legend(fontsize=9, loc='lower right')
    for bar, val in zip(bars, ils_vals):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=9, fontweight='bold')

    # ── Шкала интерпретации ────────────────────────────────────
    ax2 = axes[1]
    ax2.axis('off')
    ax2.set_title('Интерпретация ILS', fontsize=11, fontweight='bold')

    zones = [
        (0.0, 0.2,  C_RED,   'Критично: дубли\n(модель повторяет один аромат)'),
        (0.2, 0.3,  C_AMBER, 'Низкое разнообразие\n(очень похожие ароматы)'),
        (0.3, 0.5,  C_GREEN, '✓ Идеальная зона\n(похожие, но разные ароматы)'),
        (0.5, 0.7,  C_AMBER, 'Высокое разнообразие\n(слабая связь с запросом)'),
        (0.7, 1.0,  C_RED,   'Хаос: нет связи\n(случайные рекомендации)'),
    ]

    for i, (lo, hi, color, label) in enumerate(zones):
        y = 0.82 - i * 0.16
        is_current = lo <= mean_ils <= hi

        rect = mpatches.FancyBboxPatch(
            (0.02, y - 0.07), 0.96, 0.14,
            boxstyle='round,pad=0.01',
            facecolor=color + '33',
            edgecolor=color, linewidth=2 if is_current else 0.5,
            transform=ax2.transAxes, clip_on=False
        )
        ax2.add_patch(rect)

        ax2.text(0.08, y + 0.01, f'{lo:.1f} – {hi:.1f}',
                 transform=ax2.transAxes, fontsize=9.5,
                 fontweight='bold', color=color, va='center')
        ax2.text(0.28, y + 0.01, label,
                 transform=ax2.transAxes, fontsize=8.5,
                 color=C_DARK, va='center')

        if is_current:
            ax2.text(0.92, y + 0.01, f'◄ {mean_ils:.3f}',
                     transform=ax2.transAxes, fontsize=9,
                     fontweight='bold', color=color, va='center', ha='right')

    plt.tight_layout()
    out = OUTPUT_DIR / 'metric4_intra_list_similarity.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  ✓ {out}')


# ─────────────────────────────────────────────────────────────────
# ГРАФИК 5: Reconstruction Error и объяснённая дисперсия
# ─────────────────────────────────────────────────────────────────

def plot_reconstruction(data):
    rec = data.get('reconstruction_error', {})
    exp_per_comp = rec.get('explained_per_component', [])
    sing_vals    = rec.get('singular_values', [])

    if not exp_per_comp:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('white')
    fig.suptitle(
        f'Метрика Reconstruction Error — качество SVD сжатия\n'
        f'n_components={rec.get("n_components","?")}, '
        f'ошибка восстановления={rec.get("reconstruction_error","?")}, '
        f'объяснённая дисперсия={rec.get("explained_variance_ratio","?")}%',
        fontsize=12, fontweight='bold', color=C_DARK
    )

    x = range(1, len(exp_per_comp) + 1)

    # ── Накопленная объяснённая дисперсия ─────────────────────
    ax1 = axes[0]
    cumulative = np.cumsum(exp_per_comp)
    ax1.plot(x, cumulative, 'o-', color=C_BLUE, linewidth=2, markersize=5)
    ax1.fill_between(x, cumulative, alpha=0.15, color=C_BLUE)
    ax1.axhline(80, color=C_GREEN, linestyle='--', alpha=0.6, linewidth=1.5,
                label='80% дисперсии')
    ax1.axhline(90, color=C_AMBER, linestyle='--', alpha=0.6, linewidth=1.5,
                label='90% дисперсии')
    ax1.set_xlabel('Номер SVD-компоненты', fontsize=10)
    ax1.set_ylabel('Накопленная объяснённая дисперсия, %', fontsize=10)
    ax1.set_title('Накопление объяснённой дисперсии\nпо SVD-компонентам', fontsize=11)
    ax1.set_facecolor(C_LIGHT)
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, 105)

    # Отметим точку, где достигается 80%
    for i, cum in enumerate(cumulative):
        if cum >= 80:
            ax1.axvline(i+1, color=C_GREEN, linestyle=':', alpha=0.8, linewidth=1.5)
            ax1.text(i+2, 40, f'80% при\nn={i+1}',
                     fontsize=8, color=C_GREEN)
            break

    # ── Сингулярные значения (Scree plot) ─────────────────────
    ax2 = axes[1]
    ax2.plot(x, sing_vals[:len(x)], 'o-', color=C_PURPLE,
             linewidth=2, markersize=5)
    ax2.fill_between(x, sing_vals[:len(x)], alpha=0.15, color=C_PURPLE)
    ax2.set_xlabel('Номер компоненты', fontsize=10)
    ax2.set_ylabel('Сингулярное значение', fontsize=10)
    ax2.set_title('Scree Plot — сингулярные значения SVD\n'
                  '(«локоть» показывает оптимальное n_components)', fontsize=11)
    ax2.set_facecolor(C_LIGHT)
    ax2.spines[['top', 'right']].set_visible(False)

    # Ищем «локоть»
    if len(sing_vals) >= 3:
        diffs = np.diff(sing_vals[:len(x)])
        elbow = int(np.argmin(diffs)) + 1
        ax2.axvline(elbow, color=C_RED, linestyle='--', alpha=0.7, linewidth=1.5,
                    label=f'«Локоть» ≈ {elbow}')
        ax2.legend(fontsize=9)

    plt.tight_layout()
    out = OUTPUT_DIR / 'metric5_reconstruction.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  ✓ {out}')


# ─────────────────────────────────────────────────────────────────
# ГРАФИК 6: Note Drift
# ─────────────────────────────────────────────────────────────────

def plot_note_drift(data):
    nd       = data.get('note_drift', {})
    details  = nd.get('details', [])
    mean_d   = nd.get('mean_drift', 1.0)

    if not details:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')
    fig.suptitle('Метрика Note Drift — совпадение нот запроса с рекомендацией',
                 fontsize=13, fontweight='bold', color=C_DARK)

    queries   = [d['query'][:30] for d in details]
    overlaps  = [d['note_overlap'] for d in details]
    drifts    = [d['drift'] * 100 for d in details]

    # ── Парные бары: пересечение vs дрейф ─────────────────────
    ax1 = axes[0]
    x = np.arange(len(details))
    w = 0.35
    b1 = ax1.bar(x - w/2, overlaps, w, label='Пересечение нот, %',
                 color=C_GREEN, edgecolor='white')
    b2 = ax1.bar(x + w/2, drifts,   w, label='Дрейф нот, %',
                 color=C_RED, edgecolor='white', alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(queries, rotation=25, ha='right', fontsize=8)
    ax1.set_ylabel('%')
    ax1.set_title('Пересечение и дрейф нот\nдля каждого запроса')
    ax1.set_ylim(0, 115)
    ax1.axhline(50, color=C_DARK, linestyle='--', alpha=0.3, linewidth=1)
    ax1.set_facecolor(C_LIGHT)
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.legend(fontsize=9)

    for bar, val in zip(b1, overlaps):
        if val > 3:
            ax1.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 1.5,
                     f'{val:.0f}%', ha='center', fontsize=8, fontweight='bold',
                     color=C_GREEN)

    # ── Таблица: топ-1 товар vs ноты запроса ──────────────────
    ax2 = axes[1]
    ax2.axis('off')
    ax2.set_title('Детали: запрос → топ-1 рекомендация', fontsize=11, fontweight='bold')

    col_labels = ['Запрос', 'Топ-1 товар', 'Пересечение', 'Дрейф']
    col_x = [0.01, 0.30, 0.62, 0.80]
    col_w = [0.29, 0.32, 0.18, 0.19]

    # Заголовок
    for txt, x_pos, w in zip(col_labels, col_x, col_w):
        rect = mpatches.FancyBboxPatch(
            (x_pos, 0.93), w - 0.01, 0.06,
            boxstyle='round,pad=0.005', facecolor=C_DARK,
            linewidth=0, transform=ax2.transAxes, clip_on=False
        )
        ax2.add_patch(rect)
        ax2.text(x_pos + 0.01, 0.96, txt, transform=ax2.transAxes,
                 fontsize=8.5, fontweight='bold', color='white', va='center')

    for i, d in enumerate(details):
        y = 0.87 - i * 0.115
        bg = C_LIGHT if i % 2 == 0 else 'white'
        drift_color = C_GREEN if d['drift'] < 0.5 else (C_AMBER if d['drift'] < 0.8 else C_RED)

        rect = mpatches.FancyBboxPatch(
            (0.01, y - 0.04), 0.98, 0.10,
            boxstyle='round,pad=0.005', facecolor=bg,
            edgecolor=drift_color, linewidth=0.8,
            transform=ax2.transAxes, clip_on=False
        )
        ax2.add_patch(rect)

        ax2.text(col_x[0] + 0.01, y + 0.01,
                 d['query'][:25], transform=ax2.transAxes,
                 fontsize=7.5, color=C_DARK, va='center')
        ax2.text(col_x[1] + 0.01, y + 0.01,
                 d['top1_product'][:25], transform=ax2.transAxes,
                 fontsize=7.5, color=C_DARK, va='center')
        ax2.text(col_x[2] + 0.04, y + 0.01,
                 f'{d["note_overlap"]:.0f}%', transform=ax2.transAxes,
                 fontsize=9, fontweight='bold', color=C_GREEN, va='center', ha='center')
        ax2.text(col_x[3] + 0.05, y + 0.01,
                 f'{d["drift"]*100:.0f}%', transform=ax2.transAxes,
                 fontsize=9, fontweight='bold', color=drift_color, va='center', ha='center')

    ax2.text(0.5, -0.03,
             f'Средний дрейф нот: {mean_d:.3f}  '
             f'(0 = идеально, 1 = полный дрейф)',
             transform=ax2.transAxes, ha='center', fontsize=9,
             fontweight='bold',
             color=C_GREEN if mean_d < 0.5 else C_AMBER)

    plt.tight_layout()
    out = OUTPUT_DIR / 'metric6_note_drift.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  ✓ {out}')


# ─────────────────────────────────────────────────────────────────
# ГРАФИК 7: SVD n_components сравнение
# ─────────────────────────────────────────────────────────────────

def plot_svd_comparison(data):
    svd_data   = data.get('svd_components', {})
    comparison = svd_data.get('comparison', [])
    if not comparison:
        return

    n_list    = [c['n_components']       for c in comparison]
    hr5_list  = [c['hr_at_5']            for c in comparison]
    cov_list  = [c['coverage']           for c in comparison]
    ils_list  = [c['ils']               for c in comparison]
    exp_list  = [c.get('explained_variance', 0) for c in comparison]

    current_n = data.get('meta', {}).get('n_components', None)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor('white')
    fig.suptitle('Влияние n_components SVD на качество рекомендаций\n'
                 '(выбор оптимальной размерности)',
                 fontsize=13, fontweight='bold', color=C_DARK)

    def mark_current(ax):
        if current_n in n_list:
            idx = n_list.index(current_n)
            ax.axvline(current_n, color=C_PURPLE, linestyle='--', alpha=0.7,
                       linewidth=1.5, label=f'Текущее n={current_n}')

    # ── HR@5 ────────────────────────────────────────────────────
    ax1 = axes[0, 0]
    ax1.plot(n_list, hr5_list, 'o-', color=C_GREEN, linewidth=2.5, markersize=8)
    for n, v in zip(n_list, hr5_list):
        ax1.text(n, v + 0.02, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
    mark_current(ax1)
    ax1.set_xlabel('n_components')
    ax1.set_ylabel('Hit Rate@5')
    ax1.set_title('Hit Rate@5 при разных n_components')
    ax1.set_xticks(n_list)
    ax1.set_ylim(0, 1.15)
    ax1.set_facecolor(C_LIGHT)
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.legend(fontsize=9)

    # ── Coverage ─────────────────────────────────────────────────
    ax2 = axes[0, 1]
    ax2.plot(n_list, cov_list, 's-', color=C_BLUE, linewidth=2.5, markersize=8)
    for n, v in zip(n_list, cov_list):
        ax2.text(n, v + 1.5, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')
    mark_current(ax2)
    ax2.axhline(30, color=C_RED, linestyle='--', alpha=0.5, linewidth=1, label='Порог 30%')
    ax2.set_xlabel('n_components')
    ax2.set_ylabel('Coverage, %')
    ax2.set_title('Coverage@10 при разных n_components')
    ax2.set_xticks(n_list)
    ax2.set_ylim(0, max(cov_list) * 1.3 + 10 if cov_list else 100)
    ax2.set_facecolor(C_LIGHT)
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.legend(fontsize=9)

    # ── ILS ───────────────────────────────────────────────────────
    ax3 = axes[1, 0]
    ax3.plot(n_list, ils_list, '^-', color=C_AMBER, linewidth=2.5, markersize=8)
    for n, v in zip(n_list, ils_list):
        ax3.text(n, v + 0.01, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax3.axhspan(0.3, 0.5, color=C_GREEN, alpha=0.1, label='Идеальный диапазон 0.3–0.5')
    mark_current(ax3)
    ax3.set_xlabel('n_components')
    ax3.set_ylabel('Intra-list Similarity')
    ax3.set_title('ILS (разнообразие выдачи) при разных n_components')
    ax3.set_xticks(n_list)
    ax3.set_facecolor(C_LIGHT)
    ax3.spines[['top', 'right']].set_visible(False)
    ax3.legend(fontsize=9)

    # ── Объяснённая дисперсия ─────────────────────────────────────
    ax4 = axes[1, 1]
    ax4.plot(n_list, exp_list, 'D-', color=C_PURPLE, linewidth=2.5, markersize=8)
    for n, v in zip(n_list, exp_list):
        ax4.text(n, v + 1, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')
    mark_current(ax4)
    ax4.axhline(80, color=C_GREEN, linestyle='--', alpha=0.5,
                linewidth=1, label='80% дисперсии')
    ax4.set_xlabel('n_components')
    ax4.set_ylabel('Объяснённая дисперсия, %')
    ax4.set_title('Объяснённая дисперсия TF-IDF матрицы')
    ax4.set_xticks(n_list)
    ax4.set_ylim(0, 115)
    ax4.set_facecolor(C_LIGHT)
    ax4.spines[['top', 'right']].set_visible(False)
    ax4.legend(fontsize=9)

    plt.tight_layout()
    out = OUTPUT_DIR / 'metric7_svd_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  ✓ {out}')


# ─────────────────────────────────────────────────────────────────
# ГРАФИК 8: Финальная сводная таблица всех метрик
# ─────────────────────────────────────────────────────────────────

def plot_final_summary(data):
    meta = data.get('meta', {})
    hr   = data.get('hit_rate_mrr', {})
    mrr_d = data.get('hit_rate_mrr', {})
    cov  = data.get('coverage', {})
    ils  = data.get('intra_list_similarity', {})
    rec  = data.get('reconstruction_error', {})
    nd   = data.get('note_drift', {})

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor('white')
    ax.axis('off')

    fig.text(0.5, 0.97,
             'Итоговая таблица метрик рекомендательной модели TF-IDF + SVD',
             ha='center', va='top', fontsize=14, fontweight='bold', color=C_DARK)
    fig.text(0.5, 0.92,
             f'Модель: TF-IDF (словарь {meta.get("vocab_size","?")} токенов) + '
             f'SVD ({meta.get("n_components","?")} компонент) + Cosine Similarity  |  '
             f'Каталог: {meta.get("n_products","?")} товаров  |  '
             f'Тестовых запросов: {meta.get("n_queries","?")}  |  '
             f'Дата: {meta.get("timestamp","")}',
             ha='center', va='top', fontsize=8.5, color=C_GRAY)

    rows = [
        # метрика, значение, интерпретация, оценка, цвет
        ('Hit Rate@1',
         f'{hr.get("hit_rate",{}).get("1",0):.3f}',
         'Доля запросов, где топ-1 результат — релевантный товар',
         _grade(hr.get("hit_rate",{}).get("1",0), 0.5, 0.3)),
        ('Hit Rate@5',
         f'{hr.get("hit_rate",{}).get("5",0):.3f}',
         'Доля запросов с релевантным товаром в топ-5',
         _grade(hr.get("hit_rate",{}).get("5",0), 0.7, 0.4)),
        ('Hit Rate@10',
         f'{hr.get("hit_rate",{}).get("10",0):.3f}',
         'Доля запросов с релевантным товаром в топ-10',
         _grade(hr.get("hit_rate",{}).get("10",0), 0.8, 0.5)),
        ('MRR@5',
         f'{mrr_d.get("mrr",{}).get("5",0):.3f}',
         'Средняя обратная позиция первого верного ответа (топ-5)',
         _grade(mrr_d.get("mrr",{}).get("5",0), 0.5, 0.25)),
        ('MRR@10',
         f'{mrr_d.get("mrr",{}).get("10",0):.3f}',
         'Средняя обратная позиция первого верного ответа (топ-10)',
         _grade(mrr_d.get("mrr",{}).get("10",0), 0.5, 0.25)),
        ('Coverage@10',
         f'{cov.get("coverage_at_10",0):.1f}%',
         f'Доля каталога, охваченная рекомендациями '
         f'({cov.get("unique_recommended","?")} из {cov.get("total_products","?")} товаров)',
         _grade(cov.get("coverage_at_10",0), 50, 30)),
        ('Intra-list Similarity',
         f'{ils.get("mean_ils",0):.4f}',
         'Среднее косинусное сходство внутри выдачи (идеал: 0.3–0.5)',
         _grade_ils(ils.get("mean_ils",0))),
        ('Reconstruction Error',
         f'{rec.get("reconstruction_error",0):.6f}',
         f'Ошибка восстановления TF-IDF матрицы '
         f'(объяснённая дисперсия: {rec.get("explained_variance_ratio","?")}%)',
         _grade(rec.get("explained_variance_ratio",0), 70, 50)),
        ('Note Drift',
         f'{nd.get("mean_drift",1):.3f}',
         'Средний дрейф нот (0=идеально, 1=полный дрейф)',
         _grade(1 - nd.get("mean_drift",1), 0.5, 0.2)),
    ]

    col_labels = ['Метрика', 'Значение', 'Интерпретация', 'Оценка']
    col_x      = [0.01, 0.20, 0.33, 0.91]
    col_w      = [0.19, 0.13, 0.58, 0.09]

    # Заголовок таблицы
    for lbl, x, w in zip(col_labels, col_x, col_w):
        rect = mpatches.FancyBboxPatch((x, 0.83), w - 0.01, 0.065,
                                        boxstyle='round,pad=0.005',
                                        facecolor=C_DARK, linewidth=0,
                                        transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        ax.text(x + 0.01, 0.865, lbl, transform=ax.transAxes,
                fontsize=9.5, fontweight='bold', color='white', va='center')

    for i, (metric, value, interp, (grade, color)) in enumerate(rows):
        y = 0.77 - i * 0.083
        bg = C_LIGHT if i % 2 == 0 else 'white'
        rect = mpatches.FancyBboxPatch((0.01, y - 0.035), 0.98, 0.075,
                                        boxstyle='round,pad=0.005',
                                        facecolor=bg, edgecolor=color, linewidth=1,
                                        transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        ax.text(col_x[0] + 0.01, y + 0.01, metric,
                transform=ax.transAxes, fontsize=9, fontweight='bold',
                color=C_DARK, va='center')
        ax.text(col_x[1] + 0.01, y + 0.01, value,
                transform=ax.transAxes, fontsize=10, fontweight='bold',
                color=color, va='center')
        ax.text(col_x[2] + 0.01, y + 0.01, interp,
                transform=ax.transAxes, fontsize=8, color=C_GRAY, va='center')
        ax.text(col_x[3] + 0.01, y + 0.01, grade,
                transform=ax.transAxes, fontsize=9.5, fontweight='bold',
                color=color, va='center')

    plt.tight_layout()
    out = OUTPUT_DIR / 'metric8_final_summary.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  ✓ {out}')


def _grade(value, good_threshold, ok_threshold):
    if value >= good_threshold:
        return ('✓ Хорошо', C_GREEN)
    elif value >= ok_threshold:
        return ('~ Приемлемо', C_AMBER)
    else:
        return ('✗ Низкое', C_RED)


def _grade_ils(ils):
    if 0.3 <= ils <= 0.5:
        return ('✓ Идеально', C_GREEN)
    elif 0.2 <= ils < 0.3 or 0.5 < ils <= 0.6:
        return ('~ Приемлемо', C_AMBER)
    else:
        return ('✗ Низкое', C_RED)


# ─────────────────────────────────────────────────────────────────
# Точка входа
# ─────────────────────────────────────────────────────────────────

def main():
    print('\n' + '='*60)
    print('  ВИЗУАЛИЗАЦИЯ МЕТРИК РЕКОМЕНДАТЕЛЬНОЙ МОДЕЛИ')
    print('='*60)

    data = load()
    meta = data.get('meta', {})
    print(f'\nДата тестирования : {meta.get("timestamp","—")}')
    print(f'Товаров в каталоге: {meta.get("n_products","—")}')
    print(f'Тестовых запросов : {meta.get("n_queries","—")}')
    print(f'\nСохраняю графики в {OUTPUT_DIR}/...\n')

    plot_hit_rate_mrr(data)
    plot_per_query_heatmap(data)
    plot_coverage(data)
    plot_ils(data)
    plot_reconstruction(data)
    plot_note_drift(data)
    plot_svd_comparison(data)
    plot_final_summary(data)

    print(f'\n✓ Все графики сохранены в {OUTPUT_DIR}/')
    print('\nСписок файлов:')
    for f in sorted(OUTPUT_DIR.glob('metric*.png')):
        print(f'  {f.name:<45}  {f.stat().st_size//1024} KB')
    print('='*60 + '\n')


if __name__ == '__main__':
    main()