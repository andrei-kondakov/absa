from django import template

from data.constants import sb1_baseline, sb1_best

register = template.Library()


@register.inclusion_tag('stats/tags/aspect_stats.html')
def aspect_detection_stats(stats):
    best_diff_pct = (stats['f1_macro'] - sb1_best) * 100
    baseline_diff_pct = (stats['f1_macro'] - sb1_baseline) * 100
    return {
        'sb1_baseline': sb1_baseline,
        'sb1_best': sb1_best,
        'stats': stats,
        'baseline_diff_pct': baseline_diff_pct,
        'best_diff_pct': best_diff_pct
    }
