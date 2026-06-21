"""Boid fish swarm simulation launcher."""
import pathlib, webbrowser
html = pathlib.Path(__file__).with_name('boids.html').resolve()
print('Opening boids.html …')
webbrowser.open(html.as_uri())
