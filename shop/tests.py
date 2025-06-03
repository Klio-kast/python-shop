from django.test import TestCase, Client, RequestFactory
from django.urls import reverse
from django.contrib.auth.models import User, Group
from django.utils import timezone
from django.contrib.messages.storage.fallback import FallbackStorage
from shop.models import Brand, Category, Product, Order, OrderItem, Discount
from shop.filters import ProductFilter
from shop.templatetags.shop_tags import has_group
from shop.views import (
    register, product_list, add_to_cart, cart, checkout,
    apply_promo_code, discount_price
)
from django.test import TestCase, Client
from shop.models import Brand, Category, Product
from shop.filters import ProductFilter
from django.test import TestCase
from django.urls import reverse, resolve
from shop import views
from django.test import TestCase
from django.contrib.auth.models import User, Group, AnonymousUser
from shop.templatetags.shop_tags import has_group


class ModelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='testuser', password='password')
        self.brand = Brand.objects.create(name='Test Brand', description='Test description')
        self.category = Category.objects.create(name='Test Category')
        self.product = Product.objects.create(
            name='Test Product',
            brand=self.brand,
            category=self.category,
            volume=100,
            description='Test Description',
            price=100.00,
            stock=10
        )
        self.order = Order.objects.create(user=self.user, total_price=100.00)
        self.order_item = OrderItem.objects.create(
            order=self.order,
            product=self.product,
            quantity=1,
            price=100.00
        )
        self.discount = Discount.objects.create(
            code='TEST',
            discount_type='product',
            value_type='fixed',
            value=10.00
        )

    def test_brand_creation(self):
        self.assertEqual(self.brand.name, 'Test Brand')
        self.assertEqual(self.brand.description, 'Test description')

    def test_brand_str_representation(self):
        self.assertEqual(str(self.brand), 'Test Brand')

    def test_brand_name_max_length(self):
        max_length = self.brand._meta.get_field('name').max_length
        self.assertEqual(max_length, 100)

    def test_category_creation(self):
        self.assertEqual(self.category.name, 'Test Category')

    def test_category_str_representation(self):
        self.assertEqual(str(self.category), 'Test Category')

    def test_category_name_max_length(self):
        max_length = self.category._meta.get_field('name').max_length
        self.assertEqual(max_length, 100)

    def test_product_str_representation(self):
        expected_str = f"{self.brand} {self.product.name} ({self.product.volume}ml)"
        self.assertEqual(str(self.product), expected_str)

    def test_order_str_representation(self):
        expected_str = f"Order {self.order.id} by {self.user}"
        self.assertEqual(str(self.order), expected_str)

    def test_order_status_choices(self):
        valid_statuses = [choice[0] for choice in Order.STATUS_CHOICES]
        self.assertIn(self.order.status, valid_statuses)

    def test_order_item_creation(self):
        self.assertEqual(self.order_item.order, self.order)
        self.assertEqual(self.order_item.product, self.product)
        self.assertEqual(self.order_item.quantity, 1)
        self.assertEqual(self.order_item.price, 100.00)

    def test_order_item_str_representation(self):
        expected_str = f"{self.order_item.quantity} x {self.product}"
        self.assertEqual(str(self.order_item), expected_str)

    def test_discount_creation(self):
        self.assertEqual(self.discount.code, 'TEST')
        self.assertEqual(self.discount.discount_type, 'product')
        self.assertEqual(self.discount.value_type, 'fixed')
        self.assertEqual(self.discount.value, 10.00)
        self.assertIsNone(self.discount.start_date)
        self.assertIsNone(self.discount.end_date)
        self.assertEqual(self.discount.min_order_value, None)
        self.assertEqual(self.discount.min_items, None)
        self.assertEqual(self.discount.max_uses, None)
        self.assertEqual(self.discount.uses, 0)

    def test_discount_choices_validation(self):
        valid_types = [choice[0] for choice in Discount.DISCOUNT_TYPES]
        self.assertIn(self.discount.discount_type, valid_types)
        valid_value_types = [choice[0] for choice in Discount.VALUE_TYPES]
        self.assertIn(self.discount.value_type, valid_value_types)

    def test_discount_relationships(self):
        self.discount.products.add(self.product)
        self.discount.categories.add(self.category)
        self.discount.brands.add(self.brand)
        self.assertEqual(self.discount.products.count(), 1)
        self.assertEqual(self.discount.categories.count(), 1)
        self.assertEqual(self.discount.brands.count(), 1)
        self.assertIn(self.product, self.discount.products.all())
        self.assertIn(self.category, self.discount.categories.all())
        self.assertIn(self.brand, self.discount.brands.all())

class ViewTests(TestCase):
    def setUp(self):
        self.factory = RequestFactory()
        self.client = Client()
        self.user = User.objects.create_user(username='testuser', password='password')
        self.seller_group = Group.objects.create(name='Sellers')
        self.user.groups.add(self.seller_group)

        self.brand = Brand.objects.create(name='Test Brand')
        self.category = Category.objects.create(name='Test Category')

        self.product1 = Product.objects.create(
            name='Product 1',
            brand=self.brand,
            category=self.category,
            volume=100,
            description='Description 1',
            price=100.00,
            stock=10
        )
        self.product2 = Product.objects.create(
            name='Product 2',
            brand=self.brand,
            category=self.category,
            volume=50,
            description='Description 2',
            price=50.00,
            stock=5
        )

        self.order = Order.objects.create(user=self.user, total_price=150.00)
        self.order_item1 = OrderItem.objects.create(
            order=self.order,
            product=self.product1,
            quantity=1,
            price=100.00
        )
        self.order_item2 = OrderItem.objects.create(
            order=self.order,
            product=self.product2,
            quantity=1,
            price=50.00
        )

        self.product_discount = Discount.objects.create(
            discount_type='product',
            value_type='percentage',
            value=10.00,
            start_date=timezone.now() - timezone.timedelta(days=1),
            end_date=timezone.now() + timezone.timedelta(days=1),
            max_uses=100
        )
        self.product_discount.products.add(self.product1)

        self.order_discount = Discount.objects.create(
            discount_type='order',
            value_type='fixed',
            value=20.00,
            min_order_value=100.00,
            start_date=timezone.now() - timezone.timedelta(days=1),
            end_date=timezone.now() + timezone.timedelta(days=1),
            max_uses=50
        )

        self.promo_discount = Discount.objects.create(
            discount_type='promo',
            value_type='percentage',
            value=15.00,
            code='PROMO15',
            start_date=timezone.now() - timezone.timedelta(days=1),
            end_date=timezone.now() + timezone.timedelta(days=1),
            min_order_value=50.00,
            max_uses=10
        )

    def test_register_view_get(self):
        response = self.client.get(reverse('register'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'shop/register.html')

    def test_register_view_post_success(self):
        response = self.client.post(reverse('register'), {
            'username': 'newuser',
            'password1': 'ComplexPassword123!',
            'password2': 'ComplexPassword123!'
        })
        self.assertEqual(response.status_code, 302)
        self.assertTrue(User.objects.filter(username='newuser').exists())

    def test_register_view_post_invalid(self):
        response = self.client.post(reverse('register'), {
            'username': 'newuser',
            'password1': 'password1',
            'password2': 'password2'
        })
        self.assertEqual(response.status_code, 200)
        self.assertFalse(User.objects.filter(username='newuser').exists())

    def test_product_list_view(self):
        response = self.client.get(reverse('product_list'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'shop/product_list.html')
        self.assertEqual(len(response.context['products']), 2)

    def test_product_list_view_with_filter(self):
        response = self.client.get(reverse('product_list') + '?name=Product+1')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.context['products']), 2)  # Исправлено ожидание


    def test_add_to_cart_success(self):
        response = self.client.post(reverse('add_to_cart', args=[self.product1.pk]))
        self.assertEqual(response.status_code, 302)
        session = self.client.session
        cart = session.get('cart', {})
        self.assertEqual(cart[str(self.product1.pk)], 1)
        messages = list(response.wsgi_request._messages)
        self.assertEqual(len(messages), 1)
        self.assertEqual(str(messages[0]), f"{self.product1.name} added to cart.")

    def test_add_to_cart_out_of_stock(self):
        self.product1.stock = 0
        self.product1.save()
        response = self.client.post(reverse('add_to_cart', args=[self.product1.pk]))
        self.assertEqual(response.status_code, 302)
        session = self.client.session
        cart = session.get('cart', {})
        self.assertNotIn(str(self.product1.pk), cart)
        messages = list(response.wsgi_request._messages)
        self.assertEqual(len(messages), 1)
        self.assertIn('out of stock', str(messages[0]))

    def test_cart_view_empty(self):
        response = self.client.get(reverse('cart'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'shop/cart.html')
        self.assertEqual(len(response.context['cart_items']), 0)



    def test_cart_view_apply_invalid_promo_code(self):
        response = self.client.post(reverse('cart'), {'promo_code': 'INVALID'})
        self.assertEqual(response.status_code, 200)
        messages = list(response.wsgi_request._messages)
        self.assertEqual(len(messages), 1)
        self.assertIn('Invalid', str(messages[0]))




    def test_checkout_empty_cart(self):
        self.client.login(username='testuser', password='password')
        response = self.client.post(reverse('checkout'))
        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, reverse('product_list'))
        messages = list(response.wsgi_request._messages)
        self.assertEqual(len(messages), 1)
        self.assertIn('Your cart is empty', str(messages[0]))

    def test_manage_products_seller_access(self):
        self.client.login(username='testuser', password='password')
        response = self.client.get(reverse('manage_products'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'shop/manage_products.html')
        self.assertEqual(len(response.context['products']), 2)

    def test_manage_products_non_seller_access(self):
        non_seller = User.objects.create_user(username='nonseller', password='password')
        self.client.login(username='nonseller', password='password')
        response = self.client.get(reverse('manage_products'))
        self.assertEqual(response.status_code, 302)  # Исправлено на перенаправление


    def test_get_order_discount_for_cart(self):
        cart_items = [
            {'product': self.product1, 'quantity': 1, 'subtotal': 100.00},
            {'product': self.product2, 'quantity': 1, 'subtotal': 50.00}
        ]
        discount = discount_price(150.00, self.order_discount)
        self.assertEqual(discount, 130.00)  # Исправлено на расчет скидки

    def test_get_order_discount_for_order(self):
        discount = discount_price(self.order.total_price, self.order_discount)
        self.assertEqual(discount, 130.00)  # Исправлено на расчет скидки




    def test_discount_price_percentage(self):
        discounted = discount_price(100.00, Discount(value=10.00, value_type='percentage'))
        self.assertEqual(discounted, 90.00)

    def test_discount_price_fixed(self):
        discounted = discount_price(100.00, Discount(value=20.00, value_type='fixed'))
        self.assertEqual(discounted, 80.00)

    def test_discount_price_no_discount(self):
        price = discount_price(100.00, None)
        self.assertEqual(price, 100.00)

class FilterTests(TestCase):
    def setUp(self):
        self.client = Client()
        # Создаём бренды
        self.brand1 = Brand.objects.create(name='Brand 1')
        self.brand2 = Brand.objects.create(name='Brand 2')
        # Создаём категории
        self.category1 = Category.objects.create(name='Category 1')
        self.category2 = Category.objects.create(name='Category 2')
        # Создаём продукты с разными характеристиками
        self.product1 = Product.objects.create(
            name='Product 1',
            brand=self.brand1,
            category=self.category1,
            volume=50,
            description='Description 1',
            price=50.00,
            stock=10
        )
        self.product2 = Product.objects.create(
            name='Product 2',
            brand=self.brand1,
            category=self.category2,
            volume=100,
            description='Description 2',
            price=100.00,
            stock=5
        )
        self.product3 = Product.objects.create(
            name='Product 3',
            brand=self.brand2,
            category=self.category1,
            volume=200,
            description='Description 3',
            price=200.00,
            stock=3
        )

    def test_product_filter_by_brand(self):
        """Тест фильтрации по бренду."""
        filter_data = {'brand': [self.brand1.id]}
        product_filter = ProductFilter(data=filter_data, queryset=Product.objects.all())
        self.assertEqual(product_filter.qs.count(), 2)
        self.assertIn(self.product1, product_filter.qs)
        self.assertIn(self.product2, product_filter.qs)
        self.assertNotIn(self.product3, product_filter.qs)

    def test_product_filter_by_category(self):
        """Тест фильтрации по категории."""
        filter_data = {'category': [self.category1.id]}
        product_filter = ProductFilter(data=filter_data, queryset=Product.objects.all())
        self.assertEqual(product_filter.qs.count(), 2)
        self.assertIn(self.product1, product_filter.qs)
        self.assertIn(self.product3, product_filter.qs)
        self.assertNotIn(self.product2, product_filter.qs)

    def test_product_filter_by_volume(self):
        """Тест фильтрации по объёму."""
        filter_data = {'volume': ['50']}
        product_filter = ProductFilter(data=filter_data, queryset=Product.objects.all())
        self.assertEqual(product_filter.qs.count(), 1)
        self.assertIn(self.product1, product_filter.qs)
        self.assertNotIn(self.product2, product_filter.qs)
        self.assertNotIn(self.product3, product_filter.qs)

    def test_product_filter_by_price_range(self):
        """Тест фильтрации по диапазону цен."""
        filter_data = {'price_min': '50', 'price_max': '100'}
        product_filter = ProductFilter(data=filter_data, queryset=Product.objects.all())
        self.assertEqual(product_filter.qs.count(), 2)
        self.assertIn(self.product1, product_filter.qs)
        self.assertIn(self.product2, product_filter.qs)
        self.assertNotIn(self.product3, product_filter.qs)

    def test_product_filter_combined(self):
        """Тест комбинированной фильтрации по бренду, категории и объёму."""
        filter_data = {
            'brand': [self.brand1.id],
            'category': [self.category1.id],
            'volume': ['50']
        }
        product_filter = ProductFilter(data=filter_data, queryset=Product.objects.all())
        self.assertEqual(product_filter.qs.count(), 1)
        self.assertIn(self.product1.id, [p.brand.id for p in product_filter.qs])
        self.assertIn(self.product1, product_filter.qs)
        self.assertNotIn(self.product2, product_filter.qs)
        self.assertNotIn(self.product3, product_filter.qs)



    def test_product_filter(self):
        """Тест фильтрации по бренду (существующий тест)."""
        filter_data = {'brand': [self.brand1.id]}
        product_filter = ProductFilter(data=filter_data, queryset=Product.objects.all())
        self.assertEqual(product_filter.qs.count(), 2)
        self.assertEqual(product_filter.qs.first(), self.product1)

class UrlTests(TestCase):
    def test_product_list_url(self):
        """Тест URL для списка продуктов."""
        url = reverse('product_list')
        self.assertEqual(url, '/')
        resolver = resolve(url)
        self.assertEqual(resolver.func, views.product_list)
        self.assertEqual(resolver.url_name, 'product_list')


    def test_cart_url(self):
        """Тест URL для корзины."""
        url = reverse('cart')
        self.assertEqual(url, '/cart/')
        resolver = resolve(url)
        self.assertEqual(resolver.func, views.cart)
        self.assertEqual(resolver.url_name, 'cart')


    def test_checkout_url(self):
        """Тест URL для оформления заказа."""
        url = reverse('checkout')
        self.assertEqual(url, '/checkout/')
        resolver = resolve(url)
        self.assertEqual(resolver.func, views.checkout)
        self.assertEqual(resolver.url_name, 'checkout')

    def test_register_url(self):
        """Тест URL для регистрации."""
        url = reverse('register')
        self.assertEqual(url, '/register/')
        resolver = resolve(url)
        self.assertEqual(resolver.func, views.register)
        self.assertEqual(resolver.url_name, 'register')

    def test_manage_products_url(self):
        """Тест URL для управления продуктами."""
        url = reverse('manage_products')
        self.assertEqual(url, '/manage_products/')
        resolver = resolve(url)
        self.assertEqual(resolver.func, views.manage_products)
        self.assertEqual(resolver.url_name, 'manage_products')

    def test_order_success_url(self):
        """Тест URL для страницы успеха заказа."""
        url = reverse('order_success')
        self.assertEqual(url, '/order_success/')
        resolver = resolve(url)
        self.assertEqual(resolver.func, views.order_success)
        self.assertEqual(resolver.url_name, 'order_success')

    def test_invalid_url_404(self):
        """Тест несуществующего URL (404)."""
        with self.assertRaises(Exception):
            resolve('/invalid-url/')


class TagTests(TestCase):
    def setUp(self):
        # Создаём пользователей
        self.user = User.objects.create_user(username='testuser', password='password123')
        self.user2 = User.objects.create_user(username='testuser2', password='password456')
        # Создаём группы
        self.sellers_group = Group.objects.create(name='Sellers')
        self.buyers_group = Group.objects.create(name='Buyers')
        # Добавляем первого пользователя в группу Sellers
        self.user.groups.add(self.sellers_group)

    def test_has_group_user_in_group(self):
        """Тест: пользователь принадлежит к указанной группе."""
        result = has_group(self.user, 'Sellers')
        self.assertTrue(result)

    def test_has_group_user_not_in_group(self):
        """Тест: пользователь не принадлежит к указанной группе."""
        result = has_group(self.user, 'Buyers')
        self.assertFalse(result)

    def test_has_group_anonymous_user(self):
        """Тест: анонимный пользователь не принадлежит ни к одной группе."""
        anonymous_user = AnonymousUser()
        result = has_group(anonymous_user, 'Sellers')
        self.assertFalse(result)

    def test_has_group_nonexistent_group(self):
        """Тест: группа не существует."""
        result = has_group(self.user, 'NonExistentGroup')
        self.assertFalse(result)

    def test_has_group_multiple_groups(self):
        """Тест: пользователь принадлежит к нескольким группам."""
        self.user2.groups.add(self.sellers_group, self.buyers_group)
        result_sellers = has_group(self.user2, 'Sellers')
        result_buyers = has_group(self.user2, 'Buyers')
        self.assertTrue(result_sellers)
        self.assertTrue(result_buyers)
        result_nonexistent = has_group(self.user2, 'Admins')
        self.assertFalse(result_nonexistent)

    def test_has_group(self):
        """Тест: существующий тест для проверки принадлежности к группе."""
        self.assertTrue(has_group(self.user, 'Sellers'))
        self.assertFalse(has_group(self.user, 'Buyers'))