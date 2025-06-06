"""
Main URL configuration for the Django project.

This module defines the root URL patterns for the web application, routing incoming HTTP requests to appropriate views or included URL modules. It also configures serving of media files during development.
"""
from django.contrib import admin
from django.urls import path, include
from django.contrib.auth import views as auth_views
from django.conf import settings
from django.conf.urls.static import static

# Define URL patterns for routing requests to views or included URL modules.
urlpatterns = [
    # Route requests to the Django admin interface.
    path('admin/', admin.site.urls),
    # Include URL patterns from the 'shop' application for the root path.
    path('', include('shop.urls')),
    # Route requests to the login view, rendering a custom template.
    path('login/', auth_views.LoginView.as_view(template_name='shop/login.html'), name='login'),
    # Route requests to the logout view, redirecting to the product list page after logout.
    path('logout/', auth_views.LogoutView.as_view(next_page='product_list'), name='logout'),
# Serve media files during development by appending static file routing.
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)