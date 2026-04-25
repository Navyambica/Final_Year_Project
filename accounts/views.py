from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import messages
from .models import UserProfile
from change_detection.models import ChangeDetectionJob

def register_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')
        role = request.POST.get('role', 'user')
        
        if password != confirm_password:
            messages.error(request, 'Passwords do not match')
            return redirect('accounts:register')
        
        if User.objects.filter(username=username).exists():
            messages.error(request, 'Username already exists')
            return redirect('accounts:register')
        
        user = User.objects.create_user(username=username, email=email, password=password)
        UserProfile.objects.create(user=user, role=role)
        
        messages.success(request, 'Registration successful! Please login.')
        return redirect('accounts:login')
    
    return render(request, 'accounts/register.html')

def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            
            # Get or create UserProfile
            profile, created = UserProfile.objects.get_or_create(
                user=user,
                defaults={'role': 'admin' if user.is_superuser else 'user'}
            )
            
            # Redirect based on role or superuser status
            if profile.role == 'admin' or user.is_superuser:
                return redirect('admin_panel:dashboard')
            else:
                return redirect('accounts:dashboard')
        else:
            messages.error(request, 'Invalid credentials')
    
    return render(request, 'accounts/login.html')

@login_required
def dashboard_view(request):
    recent_jobs = ChangeDetectionJob.objects.filter(user=request.user).order_by('-created_at')[:5]
    total_jobs = ChangeDetectionJob.objects.filter(user=request.user).count()
    completed_jobs = ChangeDetectionJob.objects.filter(user=request.user, status='completed').count()
    
    context = {
        'recent_jobs': recent_jobs,
        'total_jobs': total_jobs,
        'completed_jobs': completed_jobs,
    }
    return render(request, 'accounts/dashboard.html', context)

def logout_view(request):
    logout(request)
    return redirect('accounts:login')
