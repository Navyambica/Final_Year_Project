from django.contrib import admin
from .models import UserProfile

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'role', 'organization', 'created_at']
    list_filter = ['role', 'created_at']
    search_fields = ['user__username', 'organization']
