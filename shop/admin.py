from django.contrib import admin
from .models import Brand, Category, Product, Order, OrderItem, Discount
from django.db.models import Sum

class OrderAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'status', 'total_price', 'discount_applied', 'promo_code', 'created_at')
    list_filter = ('status', 'created_at', 'promo_code')
    def total_sales(self, request):
        return Order.objects.aggregate(total=Sum('total_price'))['total'] or 0
    def discount_usage(self, request):
        return Discount.objects.values('code', 'discount_type').annotate(uses=Sum('uses'))
    def changelist_view(self, request, extra_context=None):
        extra_context = extra_context or {}
        extra_context['total_sales'] = self.total_sales(request)
        extra_context['discount_usage'] = self.discount_usage(request)
        return super().changelist_view(request, extra_context)

admin.site.register(Brand)
admin.site.register(Category)
admin.site.register(Product)
admin.site.register(Order, OrderAdmin)
admin.site.register(OrderItem)
admin.site.register(Discount)