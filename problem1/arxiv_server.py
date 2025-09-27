#!/usr/bin/env python3
import json
import sys
import re
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from datetime import datetime

# ------------------------
# Load data
# ------------------------
PAPERS_FILE = "./sample_data/papers.json"
CORPUS_FILE = "./sample_data/corpus_analysis.json"

papers = []
corpus_stats = {}

def load_data():
    global papers, corpus_stats
    try:
        with open(PAPERS_FILE, "r", encoding="utf-8") as f:
            papers = json.load(f)
    except FileNotFoundError:
        papers = []
    try:
        with open(CORPUS_FILE, "r", encoding="utf-8") as f:
            corpus_stats = json.load(f)
    except FileNotFoundError:
        corpus_stats = {}

# ------------------------
# HTTP Request Handler
# ------------------------
class ArxivHandler(BaseHTTPRequestHandler):
    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode("utf-8"))

    def _log(self, path, status, extra=""):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if "(0 results)" in extra:
            print(f"[{now}] {self.command} {path} - {status} -> NO RESULTS")
        else:
            print(f"[{now}] {self.command} {path} - {status} {extra}")

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)

            # /papers
            if path == "/papers":
                result = [
                    {
                        "arxiv_id": p["arxiv_id"],
                        "title": p["title"],
                        "authors": p["authors"],
                        "categories": p["categories"],
                    }
                    for p in papers
                ]
                self._send_json(result)
                self._log(path, "200 OK", f"({len(result)} results)")
                return

            # /papers/{arxiv_id}
            if path.startswith("/papers/"):
                arxiv_id = path.split("/")[-1]
                paper = next((p for p in papers if p["arxiv_id"] == arxiv_id), None)
                if paper:
                    self._send_json(paper)
                    self._log(path, "200 OK")
                else:
                    self._send_json({"error": "Paper not found"}, 404)
                    self._log(path, "404 Not Found")
                return

            # /search
            if path == "/search":
                if "q" not in query or not query["q"][0].strip():
                    self._send_json({"error": "Missing or invalid query"}, 400)
                    self._log(path, "400 Bad Request")
                    return

                q_terms = query["q"][0].lower().split()
                results = []
                for p in papers:
                    score = 0
                    matches_in = []
                    title_text = p["title"].lower()
                    abstract_text = p.get("abstract", "").lower()

                    for term in q_terms:
                        title_count = title_text.count(term)
                        abs_count = abstract_text.count(term)
                        if title_count > 0:
                            matches_in.append("title")
                        if abs_count > 0:
                            matches_in.append("abstract")
                        score += title_count + abs_count

                    if score > 0:
                        results.append({
                            "arxiv_id": p["arxiv_id"],
                            "title": p["title"],
                            "match_score": score,
                            "matches_in": list(set(matches_in))
                        })

                response = {"query": " ".join(q_terms), "results": results}
                self._send_json(response)
                self._log(path, "200 OK", f"({len(results)} results)")
                return

            # /stats
            if path == "/stats":
                if not corpus_stats:
                    self._send_json({"error": "Corpus statistics unavailable"}, 500)
                    self._log(path, "500 Internal Server Error")
                    return
                self._send_json(corpus_stats)
                self._log(path, "200 OK")
                return

            # Invalid endpoint
            self._send_json({"error": "Endpoint not found"}, 404)
            self._log(path, "404 Not Found")

        except Exception as e:
            self._send_json({"error": str(e)}, 500)
            self._log(self.path, "500 Internal Server Error")

# ------------------------
# Main
# ------------------------
def run_server(port=8080):
    load_data()
    server_address = ("", port)
    httpd = HTTPServer(server_address, ArxivHandler)
    print(f"Server running on port {port}...")
    httpd.serve_forever()

if __name__ == "__main__":
    port = 8080
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    run_server(port)
