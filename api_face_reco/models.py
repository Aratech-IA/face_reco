from django.db import models
import secrets

from django.contrib.postgres.fields import ArrayField

class Person(models.Model):
    token = models.CharField(max_length=200, unique=True)
    identity_server_group = models.CharField(max_length=200, default='')


class Tensor(models.Model):
    person = models.ForeignKey(Person, on_delete=models.CASCADE)
    encodedface = ArrayField(models.FloatField())
    md5hash = models.CharField(max_length=32, unique=True, default='')
