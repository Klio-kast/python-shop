"""
Template tags for the shop application.
"""
from django import template

register = template.Library()


@register.filter
def has_group(user, group_name):
    """Check if a user is a member of a specified group."""
    return user.groups.filter(name=group_name).exists()


@register.simple_tag(takes_context=True)
def page_url(context, page_num):
    """
    Build a pagination URL preserving existing GET parameters
    but replacing the 'page' parameter.
    Usage: {% page_url 3 %}
    """
    request = context.get('request')
    if request:
        params = request.GET.copy()
        params['page'] = str(page_num)
        return '?' + params.urlencode()
    return f'?page={page_num}'