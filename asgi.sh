#!/bin/bash

uvicorn --ws auto  --host 0.0.0.0 --port 9090 --reload projectObj.asgi:application
