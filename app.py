from flask import Flask, request, jsonify, send_from_directory, Response
import generator as gen
import time
import os

app = Flask(__name__, static_folder="static")

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/sample-rate")
def sample_rate():
    return jsonify({"sample_rate": gen.sample_rate})

@app.route("/api/export", methods=["POST"])
def export():
    data = request.json
    gen.duration = float(data.get("duration", 1.0))

    wind_on               = data.get("wind_on", True)
    wind_intensity        = float(data.get("wind_intensity", 0.5))
    wind_depth            = float(data.get("wind_depth", 0.5))
    wind_volume           = float(data.get("wind_volume", 0.5))

    thunder_on            = data.get("thunder_on", False)
    thunder_volume        = float(data.get("thunder_volume", 0.5))

    rain_on               = data.get("rain_on", True)
    rain_intensity        = float(data.get("rain_intensity", 0.5))
    rain_depth            = float(data.get("rain_depth", 0.5))
    rain_volume           = float(data.get("rain_volume", 0.5))

    ocean_on              = data.get("ocean_on", False)
    ocean_intensity       = float(data.get("ocean_intensity", 0.5))
    ocean_volume          = float(data.get("ocean_volume", 0.5))

    leaves_on             = data.get("leaves_on", False)
    leaves_volume         = float(data.get("leaves_volume", 0.5))

    buzzing_on            = data.get("buzzing_on", False)
    buzzing_intensity     = float(data.get("buzzing_intensity", 0.5))
    buzzing_volume        = float(data.get("buzzing_volume", 0.5))

    machinery_on          = data.get("machinery_on", False)
    machinery_intensity   = float(data.get("machinery_intensity", 0.5))
    machinery_volume      = float(data.get("machinery_volume", 0.5))

    cars_on               = data.get("cars_on", False)
    cars_volume           = float(data.get("cars_volume", 0.5))

    cricket_on            = data.get("cricket_on", True)
    cricket_volume        = float(data.get("cricket_volume", 0.5))

    bees_on               = data.get("bees_on", False)
    bees_volume           = float(data.get("bees_volume", 0.5))

    file_name             = data.get("file_name", "output")
    fmt                   = data.get("format", "wav")

    import numpy as np
    n = int(gen.sample_rate * gen.duration)
    sound = np.zeros(n)

    if wind_on:
        sound += gen.wind(wind_depth, wind_intensity, wind_volume)
    if thunder_on:
        sound += gen.thunder(thunder_volume)
    if rain_on:
        sound += gen.rain(rain_depth, rain_intensity, rain_volume)
    if ocean_on:
        sound += gen.ocean(ocean_intensity, ocean_volume)
    if leaves_on:
        sound += gen.leaves(leaves_volume)
    if buzzing_on:
        sound += gen.buzzing_electronics(buzzing_intensity, buzzing_intensity, buzzing_volume) # adjusted to be width
    if machinery_on:
        sound += gen.grumbling_machinery(0.0, machinery_intensity, machinery_volume)
    if cars_on:
        sound += gen.faraway_cars(cars_volume)
    if cricket_on:
        sound += gen.crickets(cricket_volume)
    if bees_on:
        sound += gen.bees(bees_volume)

    t0 = time.perf_counter()
    audio_bytes, mime_type = gen.export_to_bytes(sound, format=fmt)
    elapsed = (time.perf_counter() - t0) * 1000

    ext_map = {'wav': 'wav', 'ogg': 'ogg', 'opus': 'opus', 'mp3': 'mp3'}
    ext = ext_map.get(fmt, fmt)
    safe_name = (file_name or "output").replace("/", "_").replace("\\", "_")
    disposition = f'attachment; filename="{safe_name}.{ext}"'

    return Response(
        audio_bytes,
        status=200,
        mimetype=mime_type,
        headers={
            "Content-Disposition": disposition,
            "X-Elapsed-Ms": str(round(elapsed, 1)),
        }
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)
