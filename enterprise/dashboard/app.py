# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 red1239109-cmd
from flask import Flask, request, Response
from src.timeline import IncidentTimeline
from src.incident import IncidentRegistry
from .render_incidents import render_incidents_html
from .render_timeline import render_timeline_html

def create_app(timeline: IncidentTimeline, registry: IncidentRegistry):
    app = Flask(__name__)

    @app.route("/incidents")
    @app.route("/")
    def incidents():
        data = registry.summary(request.host.split(':')[0], 8080)
        return Response(render_incidents_html(data), mimetype="text/html")

    @app.route("/timeline")
    def timeline_view():
        iid = request.args.get("incident_id")
        events = timeline.list_recent(limit=200, incident_id=iid)
        return Response(render_timeline_html(events, iid), mimetype="text/html")

    return app
