"""
Views for the shop application.
"""
from django.core.paginator import Paginator
from django.db.models import F, Q
from django.utils import timezone
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib import messages
from .models import Product, Order, OrderItem, Discount
from .filters import ProductFilter


def is_seller(user):
    return user.groups.filter(name='Sellers').exists()


def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('product_list')
    else:
        form = UserCreationForm()
    return render(request, 'shop/register.html', {'form': form})


def product_list(request):
    queryset = Product.objects.select_related('brand', 'category').order_by('brand__name', 'name')
    product_filter = ProductFilter(request.GET, queryset=queryset)
    paginator = Paginator(product_filter.qs, 25)
    page_obj = paginator.get_page(request.GET.get('page'))
    return render(request, 'shop/product_list.html', {
        'products': page_obj,
        'filter': product_filter,
        'is_paginated': paginator.num_pages > 1,
        'page_obj': page_obj,
    })


# ─────────────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────────────

def _attach_ingredients_list(product):
    """Безопасно создаёт product.ingredients_list из list или str."""
    ingredients = getattr(product, 'ingredients', None)
    if isinstance(ingredients, list):
        product.ingredients_list = [str(i).strip() for i in ingredients if i]
    elif isinstance(ingredients, str) and ingredients.strip():
        product.ingredients_list = [i.strip() for i in ingredients.split(',') if i.strip()]
    else:
        product.ingredients_list = []
    return product


def _pks_to_products(pairs: list[tuple[int, float]]) -> list[dict]:
    """
    Принимает [(pk, score), …], загружает Product объекты одним запросом.
    Возвращает [{'product': Product, 'score': float}, …].
    """
    if not pairs:
        return []
    pk_list   = [pk for pk, _ in pairs]
    score_map = {pk: round(s * 100, 1) for pk, s in pairs}
    qs        = Product.objects.select_related('brand', 'category').filter(pk__in=pk_list)
    by_pk     = {p.pk: p for p in qs}
    return [
        {'product': by_pk[pk], 'score': score_map[pk]}
        for pk in pk_list if pk in by_pk
    ]


def _load_model_safe():
    """Возвращает (model, error_str). model=None при любой проблеме."""
    try:
        from shop.recommender import load_model
        model = load_model()
        if model is None:
            return None, (
                'Модель не обучена или устарела. '
                'Запустите: python manage.py train_recommender'
            )
        return model, None
    except ImportError:
        return None, 'Установите scikit-learn: pip install scikit-learn --timeout 120'
    except Exception as e:
        return None, f'Ошибка загрузки модели: {e}'


def _load_similar_products(product_pk: int, top_n: int = 6) -> list:
    model, _ = _load_model_safe()
    if model is None:
        return []
    try:
        from shop.recommender import get_similar_pks
        return _pks_to_products(get_similar_pks(product_pk, model, top_n=top_n))
    except Exception:
        return []


# ─────────────────────────────────────────────────────
# Views
# ─────────────────────────────────────────────────────

def product_detail(request, pk):
    product = get_object_or_404(
        Product.objects.select_related('brand', 'category'), pk=pk
    )
    _attach_ingredients_list(product)
    discount    = get_product_discount(product)
    final_price = discount_price(product.price, discount)
    similar     = _load_similar_products(product.pk, top_n=6)

    return render(request, 'shop/product_detail.html', {
        'product':     product,
        'final_price': final_price,
        'similar':     similar,
    })


def recommend(request):
    query   = request.GET.get('q', '').strip()
    results = []
    error   = None

    if query:
        model, error = _load_model_safe()
        if model is not None:
            try:
                from shop.recommender import get_pks_by_query
                top_n   = min(int(request.GET.get('top_n', 12)), 50)
                pairs   = get_pks_by_query(query, model, top_n=top_n)
                results = _pks_to_products(pairs)
            except Exception as e:
                error = f'Ошибка поиска: {e}'

    return render(request, 'shop/recommend.html', {
        'query':   query,
        'results': results,
        'error':   error,
    })


def add_to_cart(request, pk):
    product = get_object_or_404(Product, pk=pk)
    if product.stock < 1:
        messages.error(request, f"{product.name} is out of stock.")
        return redirect('product_list')
    cart = request.session.get('cart', {})
    cart[str(pk)] = cart.get(str(pk), 0) + 1
    request.session['cart'] = cart
    messages.success(request, f"{product.name} added to cart.")
    return redirect('cart')


def cart(request):
    cart_session = request.session.get('cart', {})
    cart_items   = []
    total        = 0
    for pk, qty in cart_session.items():
        product = Product.objects.get(pk=pk)
        if product.stock < qty:
            cart_session[str(pk)] = product.stock
            request.session['cart'] = cart_session
            messages.warning(request, f"Adjusted quantity for {product.name} due to stock limits.")
        discount = get_product_discount(product)
        price    = discount_price(product.price, discount)
        subtotal = price * qty
        total   += subtotal
        cart_items.append({'product': product, 'quantity': qty,
                           'subtotal': subtotal, 'discount': discount})
    order_discount = get_order_discount(cart_items)
    final_total    = total - (order_discount or 0)
    if request.method == 'POST':
        promo_code = request.POST.get('promo_code')
        if promo_code:
            discount_val = apply_promo_code(promo_code, total, cart_items)
            if discount_val:
                final_total -= discount_val
                messages.success(request, f"Promo code {promo_code} applied!")
            else:
                messages.error(request, "Invalid or expired promo code.")
    return render(request, 'shop/cart.html',
                  {'cart_items': cart_items, 'total': total, 'final_total': final_total})


@login_required
def checkout(request):
    cart_session = request.session.get('cart', {})
    if not cart_session:
        messages.error(request, "Your cart is empty.")
        return redirect('product_list')
    order = Order.objects.create(user=request.user, total_price=0)
    total = 0
    for pk, qty in cart_session.items():
        product = Product.objects.get(pk=pk)
        if product.stock < qty:
            messages.error(request, f"Not enough stock for {product.name}.")
            order.delete()
            return redirect('cart')
        discount = get_product_discount(product)
        price    = discount_price(product.price, discount)
        OrderItem.objects.create(order=order, product=product, quantity=qty, price=price)
        product.stock -= qty
        product.save()
        total += price * qty
    order_discount         = get_order_discount(order)
    order.total_price      = total
    order.discount_applied = order_discount or 0
    if request.session.get('promo_code'):
        order.promo_code = request.session['promo_code']
    order.save()
    request.session['cart'] = {}
    request.session.pop('promo_code', None)
    messages.success(request, "Order placed successfully!")
    return redirect('order_success')


# ─────────────────────────────────────────────────────
# Скидки
# ─────────────────────────────────────────────────────

def get_product_discount(product):
    now = timezone.now()
    discounts = Discount.objects.filter(
        discount_type='product', start_date__lte=now, end_date__gte=now,
    ).filter(
        Q(max_uses__isnull=True) | Q(uses__lt=F('max_uses'))
    ).filter(
        Q(products=product) | Q(categories=product.category) | Q(brands=product.brand)
    )
    return max(discounts, key=lambda d: d.value, default=None)


def get_order_discount(cart_items_or_order):
    now = timezone.now()
    discounts = Discount.objects.filter(
        discount_type='order', start_date__lte=now, end_date__gte=now,
    ).filter(Q(max_uses__isnull=True) | Q(uses__lt=F('max_uses')))
    if isinstance(cart_items_or_order, list):
        total       = sum(i['subtotal'] for i in cart_items_or_order)
        items_count = sum(i['quantity'] for i in cart_items_or_order)
    else:
        total       = cart_items_or_order.total_price
        items_count = cart_items_or_order.orderitem_set.count()
    best, max_val = None, 0
    for d in discounts:
        if (d.min_order_value and total < d.min_order_value) or \
           (d.min_items and items_count < d.min_items):
            continue
        v = d.value if d.value_type == 'fixed' else total * d.value / 100
        if v > max_val:
            max_val, best = v, d
    return max_val if best else None


def apply_promo_code(code, total, cart_items):
    now = timezone.now()
    try:
        d = Discount.objects.filter(
            code=code, discount_type='promo',
            start_date__lte=now, end_date__gte=now,
        ).filter(Q(max_uses__isnull=True) | Q(uses__lt=F('max_uses'))).first()
        if not d:
            return None
        items_count = sum(i['quantity'] for i in cart_items)
        if (d.min_order_value and total < d.min_order_value) or \
           (d.min_items and items_count < d.min_items):
            return None
        value = d.value if d.value_type == 'fixed' else total * d.value / 100
        d.uses += 1
        d.save()
        return value
    except Discount.DoesNotExist:
        return None


def discount_price(price, discount):
    if not discount:
        return price
    return price * (1 - discount.value / 100) if discount.value_type == 'percentage' else price - discount.value


@login_required
@user_passes_test(is_seller)
def manage_products(request):
    products = Product.objects.select_related('brand', 'category').all()
    return render(request, 'shop/manage_products.html', {'products': products})


def order_success(request):
    return render(request, 'shop/order_success.html')