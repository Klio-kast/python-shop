"""
Template tags for the shop application.

This module defines custom template filters for use in Django templates, enabling additional functionality for rendering dynamic content.
"""
from django import template

# Initialize the template library for registering custom tags and filters.
register = template.Library()

# Register a filter to check if a user belongs to a specific group.
@register.filter
def has_group(user, group_name):
    """
    Check if a user is a member of a specified group.

    Returns True if the user belongs to the group with the given name, False otherwise. This filter is useful for conditionally rendering template content based on user group membership.
    """
    return user.groups.filter(name=group_name).exists()