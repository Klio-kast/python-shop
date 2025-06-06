"""
URL configuration for the shop application.

This module defines the URL patterns for the shop application, mapping URLs to views for handling product listings, cart operations, user registration, and order processing.
"""
from django.urls import path
from . import views

# Define URL patterns for routing requests to views.
urlpatterns = [
    # Route requests to the product list view.
    path('', views.product_list, name='product_list'),
    # Route requests to the product detail view for a specific product.
    path('product/<int:pk>/', views.product_detail, name='product_detail'),
    # Route requests to the cart view for displaying and managing the shopping cart.
    path('cart/', views.cart, name='cart'),
    # Route requests to add a specific product to the cart.
    path('add_to_cart/<int:pk>/', views.add_to_cart, name='add_to_cart'),
    # Route requests to the checkout view for processing orders.
    path('checkout/', views.checkout, name='checkout'),
    # Route requests to the user registration view.
    path('register/', views.register, name='register'),
    # Route requests to the product management view for authorized users.
    path('manage_products/', views.manage_products, name='manage_products'),
    # Route requests to the order success view after completing an order.
    path('order_success/', views.order_success, name='order_success'),
]