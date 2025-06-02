import django_filters
from .models import Product, Brand, Category

class ProductFilter(django_filters.FilterSet):
    brand = django_filters.ModelMultipleChoiceFilter(queryset=Brand.objects.all())
    category = django_filters.ModelMultipleChoiceFilter(queryset=Category.objects.all())
    volume = django_filters.MultipleChoiceFilter(choices=[(50, '50ml'), (100, '100ml'), (200, '200ml')])
    price = django_filters.RangeFilter()
    class Meta:
        model = Product
        fields = ['brand', 'category', 'volume', 'price']