"""
Admin configuration for the shop application.

This module registers models with the Django admin interface and customizes the admin view for the Order model to display additional statistics, such as total sales and discount usage.
"""
from django.contrib import admin
from django.db.models import Sum
from .models import Brand, Category, Product, Order, OrderItem, Discount

# Define a custom admin interface for the Order model.
class OrderAdmin(admin.ModelAdmin):
    """
    Custom admin interface for managing Order instances.

    Provides a customized list view with filters and additional context for displaying total sales and discount usage statistics.
    """
    # Specify fields to display in the admin list view.
    list_display = ('id', 'user', 'status', 'total_price', 'discount_applied', 'promo_code', 'created_at')
    # Enable filtering by specified fields in the admin list view.
    list_filter = ('status', 'created_at', 'promo_code')

    def total_sales(self, request):
        """
        Calculate the total sales amount across all orders.

        Aggregates the sum of total_price for all Order instances, returning 0 if no orders exist.
        """
        return Order.objects.aggregate(total=Sum('total_price'))['total'] or 0

    def discount_usage(self, request):
        """
        Retrieve usage statistics for discounts.

        Returns a queryset with discount codes, their types, and the total number of uses for each.
        """
        return Discount.objects.values('code', 'discount_type').annotate(uses=Sum('uses'))

    def changelist_view(self, request, extra_context=None):
        """
        Customize the admin list view for Orders.

        Adds total sales and discount usage statistics to the context for display in the admin interface.
        """
        extra_context = extra_context or {}
        extra_context['total_sales'] = self.total_sales(request)
        extra_context['discount_usage'] = self.discount_usage(request)
        return super().changelist_view(request, extra_context)

# Register the Brand model with the default admin interface.
admin.site.register(Brand)
# Register the Category model with the default admin interface.
admin.site.register(Category)
# Register the Product model with the default admin interface.
admin.site.register(Product)
# Register the Order model with the customized OrderAdmin interface.
admin.site.register(Order, OrderAdmin)
# Register the OrderItem model with the default admin interface.
admin.site.register(OrderItem)
# Register the Discount model with the default admin interface.
admin.site.register(Discount)