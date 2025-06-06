"""
Configuration module for the shop application.

This module defines the configuration class for the shop application, specifying settings for model field behavior and application metadata.
"""
from django.apps import AppConfig

# Define the configuration for the shop application.
class ShopConfig(AppConfig):
    """
    Configuration class for the shop application.

    Sets the default auto-increment field type for models and specifies the application name for Django's application registry.
    """
    # Specify the default auto-increment field type for model primary keys.
    default_auto_field = 'django.db.models.BigAutoField'
    # Define the name of the application as recognized by Django.
    name = 'shop'