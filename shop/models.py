from django.db import models
from django.contrib.auth.models import User

class Brand(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    def __str__(self): return self.name

class Category(models.Model):
    name = models.CharField(max_length=100)  # e.g., Eau de Parfum, Eau de Toilette
    def __str__(self): return self.name

class Product(models.Model):
    name = models.CharField(max_length=100)
    brand = models.ForeignKey(Brand, on_delete=models.CASCADE)
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
    volume = models.PositiveIntegerField()  # in milliliters, e.g., 50, 100
    description = models.TextField()
    price = models.DecimalField(max_digits=10, decimal_places=2)
    stock = models.PositiveIntegerField(default=0)
    image = models.ImageField(upload_to='products/', null=True, blank=True)
    def __str__(self): return f"{self.brand} {self.name} ({self.volume}ml)"

class Order(models.Model):
    STATUS_CHOICES = [('Pending', 'Pending'), ('Processing', 'Processing'), ('Shipped', 'Shipped'), ('Delivered', 'Delivered')]
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='Pending')
    total_price = models.DecimalField(max_digits=10, decimal_places=2)
    discount_applied = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    promo_code = models.CharField(max_length=20, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    def __str__(self): return f"Order {self.id} by {self.user}"

class OrderItem(models.Model):
    order = models.ForeignKey(Order, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField()
    price = models.DecimalField(max_digits=10, decimal_places=2)
    def __str__(self): return f"{self.quantity} x {self.product}"

class Discount(models.Model):
    DISCOUNT_TYPES = [('product', 'Product'), ('order', 'Order'), ('promo', 'Promo Code')]
    VALUE_TYPES = [('percentage', 'Percentage'), ('fixed', 'Fixed')]
    code = models.CharField(max_length=20, unique=True, blank=True, null=True)
    discount_type = models.CharField(max_length=10, choices=DISCOUNT_TYPES)
    value_type = models.CharField(max_length=10, choices=VALUE_TYPES)
    value = models.DecimalField(max_digits=10, decimal_places=2)
    start_date = models.DateTimeField(null=True, blank=True)
    end_date = models.DateTimeField(null=True, blank=True)
    products = models.ManyToManyField(Product, blank=True)
    categories = models.ManyToManyField(Category, blank=True)
    brands = models.ManyToManyField(Brand, blank=True)
    min_order_value = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    min_items = models.PositiveIntegerField(null=True, blank=True)
    max_uses = models.PositiveIntegerField(null=True, blank=True)
    uses = models.PositiveIntegerField(default=0)
    def __str__(self): return f"{self.code or self.discount_type} - {self.value} {self.value_type}"