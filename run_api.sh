#!/usr/bin/env bash
export PYTHONPATH=.:src
uvicorn api.app.main:app --reload --port 8080
