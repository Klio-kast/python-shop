"""
Filter configuration for the shop application.

This module defines a filter set for the Product model, enabling flexible querying based on brand, category, volume, and price range in views and templates.
"""
import django_filters
from .models import Product, Brand, Category

# Define a filter set for querying Product instances.
class ProductFilter(django_filters.FilterSet):
    """
    Filter set for the Product model.

    Provides filtering capabilities for products based on multiple brands, categories, predefined volume options, and a price range. This filter set is designed for use in views to narrow down product querysets dynamically.
    """
    # Filter products by multiple brands using a dropdown selection.
    brand = django_filters.ModelMultipleChoiceFilter(queryset=Brand.objects.all())
    # Filter products by multiple categories using a dropdown selection.
    category = django_filters.ModelMultipleChoiceFilter(queryset=Category.objects.all())
    # Filter products by predefined volume options.
    volume = django_filters.MultipleChoiceFilter(choices=[(50, '50ml'), (100, '100ml'), (200, '200ml')])
    # Filter products by a range of prices.
    price = django_filters.RangeFilter()

    # Define metadata for the filter set.
    class Meta:
        """
        Metadata configuration for the ProductFilter.

        Specifies the model to filter and the fields to include in the filter set.
        """
        # Associate the filter set with the Product model.
        model = Product
        # List the fields available for filtering.
        fields = ['brand', 'category', 'volume', 'price']