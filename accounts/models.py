from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    role = models.CharField(max_length=20, choices=[('user', 'User'), ('admin', 'Admin')])
    phone = models.CharField(max_length=15, blank=True)
    organization = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.role}"


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    """Auto-create UserProfile when a User is created"""
    if created:
        # If user is a superuser, assign admin role, otherwise user role
        role = 'admin' if instance.is_superuser else 'user'
        UserProfile.objects.create(user=instance, role=role)


@receiver(post_save, sender=User)
def update_user_profile(sender, instance, **kwargs):
    """Update UserProfile role when superuser status changes"""
    try:
        profile = instance.userprofile
        if instance.is_superuser and profile.role != 'admin':
            profile.role = 'admin'
            profile.save()
    except UserProfile.DoesNotExist:
        pass
