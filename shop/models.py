"""
Models for the shop application.

This module defines the database models for managing brands, categories, products, orders, order items, and discounts in an e-commerce application.
"""
from django.db import models
from django.contrib.auth.models import User

# Define the Brand model for storing brand information.
class Brand(models.Model):
    """
    Model representing a brand in the shop.

    Stores the name and optional description of a brand, used to categorize products.
    """
    # Store the name of the brand with a maximum length of 100 characters.
    name = models.CharField(max_length=100)
    # Store an optional description of the brand, which can be empty.
    description = models.TextField(blank=True)

    def __str__(self):
        """
        Return a string representation of the brand.

        Returns the brand's name for display in admin interfaces and templates.
        """
        return self.name

# Define the Category model for organizing products.
class Category(models.Model):
    """
    Model representing a product category.

    Stores the name of a category, such as Eau de Parfum or Eau de Toilette, to group products.
    """
    # Store the name of the category with a maximum length of 100 characters.
    name = models.CharField(max_length=100)  # e.g., Eau de Parfum, Eau de Toilette

    def __str__(self):
        """
        Return a string representation of the category.

        Returns the category's name for display purposes.
        """
        return self.name

# Define the Product model for storing product details.
# shop/models.py
class Product(models.Model):
    name = models.CharField(max_length=200)
    brand = models.ForeignKey(Brand, on_delete=models.CASCADE)
    category = models.ForeignKey(Category, on_delete=models.CASCADE)

    description = models.TextField(blank=True)
    price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    stock = models.PositiveIntegerField(default=50)
    image = models.ImageField(upload_to='products/', null=True, blank=True)

    # Поля из датасета MrBob23/perfume-description
    top_notes = models.TextField(blank=True)
    middle_notes = models.TextField(blank=True)
    base_notes = models.TextField(blank=True)
    main_accords = models.JSONField(default=dict, blank=True)
    gender_ratings = models.JSONField(default=dict, blank=True)
    seasonal_ratings = models.JSONField(default=dict, blank=True)
    image_url = models.URLField(max_length=500, blank=True, null=True)  # ссылка из датасета
    volume = models.PositiveIntegerField(null=True, blank=True)
    def __str__(self):
        return f"{self.brand} — {self.name}"

    def __str__(self):
        """
        Return a string representation of the product.

        Returns a formatted string including the brand, name, and volume of the product.
        """
        return f"{self.brand} {self.name} ({self.volume}ml)"

# Define the Order model for tracking customer orders.
class Order(models.Model):
    """
    Model representing a customer order.

    Tracks order details, including the user, status, total price, applied discount, and creation timestamp.
    """
    # Define choices for the order status field.
    STATUS_CHOICES = [('Pending', 'Pending'), ('Processing', 'Processing'), ('Shipped', 'Shipped'), ('Delivered', 'Delivered')]
    # Link the order to a user, deleting the order if the user is deleted.
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    # Store the order status with predefined choices, defaulting to 'Pending'.
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='Pending')
    # Store the total price of the order with up to 10 digits and 2 decimal places.
    total_price = models.DecimalField(max_digits=10, decimal_places=2)
    # Store the discount amount applied to the order, defaulting to 0.
    discount_applied = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    # Store an optional promo code used for the order, up to 20 characters.
    promo_code = models.CharField(max_length=20, blank=True, null=True)
    # Store the timestamp when the order was created.
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        """
        Return a string representation of the order.

        Returns a formatted string including the order ID and the user's username.
        """
        return f"Order {self.id} by {self.user}"

# Define the OrderItem model for storing items within an order.
class OrderItem(models.Model):
    """
    Model representing an item in an order.

    Stores details about a product included in an order, including quantity and price at the time of purchase.
    """
    # Link the order item to an order, deleting the item if the order is deleted.
    order = models.ForeignKey(Order, on_delete=models.CASCADE)
    # Link the order item to a product, deleting the item if the product is deleted.
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    # Store the quantity of the product ordered as a positive integer.
    quantity = models.PositiveIntegerField()
    # Store the price per unit at the time of purchase with up to 10 digits and 2 decimal places.
    price = models.DecimalField(max_digits=10, decimal_places=2)

    def __str__(self):
        """
        Return a string representation of the order item.

        Returns a formatted string indicating the quantity and product details.
        """
        return f"{self.quantity} x {self.product}"

# Define the Discount model for managing discounts and promotions.
class Discount(models.Model):
    """
    Model representing a discount or promotion.

    Stores details about discounts, including type, value, applicable products, and usage limits.
    """
    # Define choices for the discount type field.
    DISCOUNT_TYPES = [('product', 'Product'), ('order', 'Order'), ('promo', 'Promo Code')]
    # Define choices for the value type field.
    VALUE_TYPES = [('percentage', 'Percentage'), ('fixed', 'Fixed')]
    # Store an optional unique discount code, up to 20 characters.
    code = models.CharField(max_length=20, unique=True, blank=True, null=True)
    # Store the type of discount with predefined choices.
    discount_type = models.CharField(max_length=10, choices=DISCOUNT_TYPES)
    # Store the type of discount value with predefined choices.
    value_type = models.CharField(max_length=10, choices=VALUE_TYPES)
    # Store the discount value with up to 10 digits and 2 decimal places.
    value = models.DecimalField(max_digits=10, decimal_places=2)
    # Store the optional start date for the discount's validity.
    start_date = models.DateTimeField(null=True, blank=True)
    # Store the optional end date for the discount's validity.
    end_date = models.DateTimeField(null=True, blank=True)
    # Link the discount to multiple products, allowing empty relations.
    products = models.ManyToManyField(Product, blank=True)
    # Link the discount to multiple categories, allowing empty relations.
    categories = models.ManyToManyField(Category, blank=True)
    # Link the discount to multiple brands, allowing empty relations.
    brands = models.ManyToManyField(Brand, blank=True)
    # Store the optional minimum order value required to apply the discount.
    min_order_value = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    # Store the optional minimum number of items required to apply the discount.
    min_items = models.PositiveIntegerField(null=True, blank=True)
    # Store the optional maximum number of times the discount can be used.
    max_uses = models.PositiveIntegerField(null=True, blank=True)
    # Store the number of times the discount has been used, defaulting to 0.
    uses = models.PositiveIntegerField(default=0)

    def __str__(self):
        """
        Return a string representation of the discount.

        Returns a formatted string including the discount code (or type if no code) and its value with type.
        """
        return f"{self.code or self.discount_type} - {self.value} {self.value_type}"