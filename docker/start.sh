#!/usr/bin/env bash
python3 /App/Api_IA/manage.py migrate > /App/log_migrate-$(date +"%Y_%m_%d_%T").log 2>&1
python3 /App/Api_IA/manage.py collectstatic --noinput > /App/log_collect_static-$(date +"%Y_%m_%d_%T").log 2>&1
env >> /etc/environment
cd Api_IA && /App/Api_IA/asgi.sh
