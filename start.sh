#!/bin/bash

# 直接執行 uvicorn，加上 access log
exec uvicorn app:app --host 0.0.0.0 --port 8005 --access-log
