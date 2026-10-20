/* ==========================================================================
   NeuroSense — waveform renderer
   --------------------------------------------------------------------------
   Draws the peak envelope of an audio file on a canvas. Used twice: when the
   user picks a file (decoded in the browser, before upload) and when showing
   the stored recording alongside a result.

   This visualises a FILE, not a microphone. The backend has no recording
   endpoint, so there is nothing to capture live into.
   ========================================================================== */

(function (global) {
    "use strict";

    var AudioCtx = global.AudioContext || global.webkitAudioContext;

    function cssVar(name, fallback) {
        var v = getComputedStyle(document.documentElement).getPropertyValue(name);
        return (v && v.trim()) || fallback;
    }

    /** Reduce samples to one [min, max] pair per pixel column. */
    function envelope(channel, columns) {
        var block = Math.floor(channel.length / columns) || 1;
        var peaks = new Float32Array(columns * 2);

        for (var c = 0; c < columns; c++) {
            var start = c * block;
            var min = 1.0, max = -1.0;
            for (var i = 0; i < block; i++) {
                var v = channel[start + i];
                if (v === undefined) break;
                if (v < min) min = v;
                if (v > max) max = v;
            }
            peaks[c * 2] = min === 1.0 ? 0 : min;
            peaks[c * 2 + 1] = max === -1.0 ? 0 : max;
        }
        return peaks;
    }

    function draw(canvas, buffer, opts) {
        var cfg = Object.assign({ animate: true }, opts || {});
        var dpr = global.devicePixelRatio || 1;
        var w = canvas.clientWidth || 600;
        var h = canvas.clientHeight || 140;

        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);

        var ctx = canvas.getContext("2d");
        ctx.scale(dpr, dpr);

        var columns = Math.max(60, Math.floor(w / 3));
        var peaks = envelope(buffer.getChannelData(0), columns);
        var mid = h / 2;
        var colWidth = w / columns;

        var ink = cssVar("--accent", "#1c5551");
        var rule = cssVar("--rule", "#ddd6c9");

        ctx.clearRect(0, 0, w, h);

        /* Centre line */
        ctx.strokeStyle = rule;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, mid);
        ctx.lineTo(w, mid);
        ctx.stroke();

        ctx.fillStyle = ink;

        function paint(upTo) {
            for (var c = 0; c < upTo; c++) {
                var min = peaks[c * 2], max = peaks[c * 2 + 1];
                var y1 = mid + min * mid * 0.92;
                var y2 = mid + max * mid * 0.92;
                var barH = Math.max(1.4, y2 - y1);
                ctx.fillRect(c * colWidth, y1, Math.max(1, colWidth - 1), barH);
            }
        }

        var reduce = global.NeuroSpiral && global.NeuroSpiral.reduceMotion.matches;
        if (!cfg.animate || reduce) { paint(columns); return; }

        /* Sweep left to right once, matching the spiral draw-in timing. */
        var startedAt = null;
        function frame(ts) {
            if (startedAt === null) startedAt = ts;
            var t = Math.min(1, (ts - startedAt) / 900);
            ctx.clearRect(0, 0, w, h);
            ctx.strokeStyle = rule;
            ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(w, mid); ctx.stroke();
            ctx.fillStyle = ink;
            paint(Math.floor(columns * t));
            if (t < 1) requestAnimationFrame(frame);
        }
        requestAnimationFrame(frame);
    }

    function fail(canvas, message) {
        var ctx = canvas.getContext("2d");
        var w = canvas.clientWidth || 600, h = canvas.clientHeight || 140;
        canvas.width = w; canvas.height = h;
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = cssVar("--ink-3", "#5c6562");
        ctx.font = '14px "IBM Plex Mono", monospace';
        ctx.textAlign = "center";
        ctx.fillText(message, w / 2, h / 2);
    }

    function render(canvas, arrayBuffer, opts) {
        if (!AudioCtx) { fail(canvas, "Waveform unavailable in this browser"); return; }
        var ctx = new AudioCtx();
        ctx.decodeAudioData(arrayBuffer)
            .then(function (buffer) {
                draw(canvas, buffer, opts);
                var meta = canvas.getAttribute("data-meta-target");
                if (meta) {
                    var el = document.getElementById(meta);
                    if (el) {
                        el.textContent = buffer.duration.toFixed(2) + " s · " +
                            buffer.sampleRate + " Hz · " +
                            buffer.numberOfChannels + " ch";
                    }
                }
            })
            .catch(function () { fail(canvas, "Could not decode this audio"); })
            .finally(function () { if (ctx.close) ctx.close(); });
    }

    global.NeuroWave = {
        fromFile: function (canvas, file, opts) {
            file.arrayBuffer().then(function (b) { render(canvas, b, opts); });
        },
        fromUrl: function (canvas, url, opts) {
            fetch(url)
                .then(function (r) { return r.arrayBuffer(); })
                .then(function (b) { render(canvas, b, opts); })
                .catch(function () { fail(canvas, "Could not load audio"); });
        }
    };
})(window);
