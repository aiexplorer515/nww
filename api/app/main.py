# FastAPI skeleton for NWW — REST + GraphQL(strawberry) mount
from fastapi import FastAPI, Depends
from fastapi.responses import JSONResponse
import os

from .routers import health, events

app = FastAPI(title="NWW API", version=os.getenv("NWW_API_VER","0.1.0"))

# Routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(events.router, prefix="/v1/events", tags=["events"])

# GraphQL endpoint (optional; requires strawberry)
try:
    import strawberry
    from strawberry.fastapi import GraphQLRouter

    @strawberry.type
    class Person:
        id: str
        canonicalName: str | None = None

    @strawberry.type
    class Event:
        id: str
        schema: str
        time: str
        place: str | None = None

    @strawberry.type
    class Query:
        person: Person | None = None
        event: Event | None = None

    schema = strawberry.Schema(Query)
    graphql_app = GraphQLRouter(schema)
    app.include_router(graphql_app, prefix="/graphql")
except Exception as e:  # pragma: no cover
    @app.get("/graphql")
    def graphql_placeholder():
        return JSONResponse({"error": "strawberry-graphql not installed", "detail": str(e)}, status_code=501)
