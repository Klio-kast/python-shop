from django.db.models import F, Value
from django.http import request
from django.utils import timezone
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.models import Group
from django.contrib import messages
from . import models
from .models import Product, Order, OrderItem, Discount
from .filters import ProductFilter


def is_seller(user):
    return user.groups.filter(name='Sellers').exists()

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            role = request.POST.get('role', 'Customer')
            group = Group.objects.get(name=role + 's')
            user.groups.add(group)
            login(request, user)
            messages.success(request, f"Welcome, {user.username}!")
            return redirect('product_list')
    else:
        form = UserCreationForm()
    return render(request, 'shop/register.html', {'form': form})

def product_list(request):
    products = Product.objects.all()
    filter = ProductFilter(request.GET, queryset=products)
    products = filter.qs
    return render(request, 'shop/product_list.html', {'products': products, 'filter': filter})

def product_detail(request, pk):
    product = get_object_or_404(Product, pk=pk)
    discount = get_product_discount(product)
    final_price = discount_price(product.price, discount)
    return render(request, 'shop/product_detail.html', {'product': product, 'final_price': final_price})

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
    cart = request.session.get('cart', {})
    cart_items = []
    total = 0
    for pk, qty in cart.items():
        product = Product.objects.get(pk=pk)
        if product.stock < qty:
            cart[str(pk)] = product.stock
            request.session['cart'] = cart
            messages.warning(request, f"Adjusted quantity for {product.name} due to stock limits.")
        discount = get_product_discount(product)
        price = discount_price(product.price, discount)
        subtotal = price * qty
        total += subtotal
        cart_items.append({'product': product, 'quantity': qty, 'subtotal': subtotal, 'discount': discount})
    order_discount = get_order_discount(cart_items)
    final_total = total - (order_discount or 0)
    if request.method == 'POST':
        promo_code = request.POST.get('promo_code')
        if promo_code:
            discount = apply_promo_code(promo_code, total, cart_items)
            if discount:
                final_total -= discount
                messages.success(request, f"Promo code {promo_code} applied!")
            else:
                messages.error(request, "Invalid or expired promo code.")
    return render(request, 'shop/cart.html', {'cart_items': cart_items, 'total': total, 'final_total': final_total})

@login_required
def checkout(request):
    cart = request.session.get('cart', {})
    if not cart:
        messages.error(request, "Your cart is empty.")
        return redirect('product_list')
    order = Order.objects.create(user=request.user, total_price=0)
    total = 0
    for pk, qty in cart.items():
        product = Product.objects.get(pk=pk)
        if product.stock < qty:
            messages.error(request, f"Not enough stock for {product.name}.")
            order.delete()
            return redirect('cart')
        discount = get_product_discount(product)
        price = discount_price(product.price, discount)
        OrderItem.objects.create(order=order, product=product, quantity=qty, price=price)
        product.stock -= qty
        product.save()
        total += price * qty
    order_discount = get_order_discount(order)
    order.total_price = total
    order.discount_applied = order_discount or 0
    if request.session.get('promo_code'):
        order.promo_code = request.session['promo_code']
    order.save()
    request.session['cart'] = {}
    request.session.pop('promo_code', None)
    messages.success(request, "Order placed successfully!")
    return redirect('order_success')

def get_product_discount(product):
    now = timezone.now()
    discounts = Discount.objects.filter(
        discount_type='product',
        start_date__lte=now,
        end_date__gte=now,
        uses__lt=models.F('max_uses') if models.F('max_uses') else models.Value(999999)
    ).filter(
        models.Q(products=product) | models.Q(categories=product.category) | models.Q(brands=product.brand)
    )
    return max(discounts, key=lambda d: d.value, default=None)

def get_order_discount(cart_items_or_order):
    now = timezone.now()
    discounts = Discount.objects.filter(
        discount_type='order',
        start_date__lte=now,
        end_date__gte=now,
        uses__lt=F('max_uses') if F('max_uses') else Value(999999)
    )
    if isinstance(cart_items_or_order, list):
        total = sum(item['subtotal'] for item in cart_items_or_order)
        items_count = sum(item['quantity'] for item in cart_items_or_order)
    else:
        total = cart_items_or_order.total_price
        items_count = cart_items_or_order.orderitem_set.count()
    best_discount = None
    max_value = 0
    for d in discounts:
        if (d.min_order_value and total < d.min_order_value) or (d.min_items and items_count < d.min_items):
            continue
        value = d.value if d.value_type == 'fixed' else total * d.value / 100
        if value > max_value:
            max_value = value
            best_discount = d
    return max_value if best_discount else None

def apply_promo_code(code, total, cart_items):
    from django.db import models
    now = timezone.now()
    try:
        discount = Discount.objects.get(
            code=code,
            discount_type='promo',
            start_date__lte=now,
            end_date__gte=now,
            uses__lt=models.F('max_uses') if models.F('max_uses') else models.Value(999999)
        )
        items_count = sum(item['quantity'] for item in cart_items)
        if (discount.min_order_value and total < discount.min_order_value) or (discount.min_items and items_count < discount.min_items):
            return None
        value = discount.value if discount.value_type == 'fixed' else total * discount.value / 100
        discount.uses += 1
        discount.save()
        request.session['promo_code'] = code
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
    products = Product.objects.all()
    return render(request, 'shop/manage_products.html', {'products': products})

def order_success(request):
    return render(request, 'shop/order_success.html')

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            print(f"User {user.username} saved successfully!")  # Отладка
            login(request, user)
            return redirect('product_list')
        else:
            print(form.errors)  # Вывод ошибок
    else:
        form = UserCreationForm()
    return render(request, 'shop/register.html', {'form': form})