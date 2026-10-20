/* ==========================================================================
   NeuroSense — Archimedes spiral engine
   --------------------------------------------------------------------------
   One generator produces every spiral in the interface: the logo mark, the
   hero's steady/tremor pair, section dividers, the loading indicator and the
   background texture. The tremor variant modulates the radius with a ~5 Hz
   oscillation (the characteristic Parkinsonian rest-tremor band) plus seeded
   smooth noise, so it is deterministic and reads as a hand, not as noise.
   ========================================================================== */

(function (global) {
    "use strict";

    var reduceMotion = global.matchMedia
        ? global.matchMedia("(prefers-reduced-motion: reduce)")
        : { matches: false };

    /* Deterministic PRNG so a given seed always draws the same hand. */
    function mulberry32(seed) {
        var t = seed >>> 0;
        return function () {
            t = (t + 0x6d2b79f5) >>> 0;
            var r = Math.imul(t ^ (t >>> 15), 1 | t);
            r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
            return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
        };
    }

    /* Smooth 1-D value noise — linear interpolation between seeded samples. */
    function smoothNoise(rand, count) {
        var anchors = [];
        for (var i = 0; i < count; i++) anchors.push(rand() * 2 - 1);
        return function (t) {
            var x = t * (count - 1);
            var i0 = Math.floor(x);
            var i1 = Math.min(i0 + 1, count - 1);
            var f = x - i0;
            var s = f * f * (3 - 2 * f); // smoothstep
            return anchors[i0] * (1 - s) + anchors[i1] * s;
        };
    }

    /**
     * Archimedes spiral path: r = a + b*theta.
     *
     * @param {Object} o
     * @param {number} o.cx,cy      centre
     * @param {number} o.turns      number of revolutions
     * @param {number} o.a          starting radius
     * @param {number} o.b          radial growth per radian
     * @param {number} o.points     sample count (higher = smoother)
     * @param {number} o.tremor     tremor amplitude in px (0 = perfectly steady)
     * @param {number} o.seed       PRNG seed
     * @param {number} o.tremorHz   oscillations per revolution (~5 Hz feel)
     * @returns {string} SVG path data
     */
    function spiralPath(o) {
        var cfg = Object.assign({
            cx: 100, cy: 100, turns: 4.25, a: 2, b: 3.4,
            points: 720, tremor: 0, seed: 7, tremorHz: 11
        }, o || {});

        var rand = mulberry32(cfg.seed);
        var noise = smoothNoise(rand, 48);
        var total = cfg.turns * Math.PI * 2;
        var d = "";

        for (var i = 0; i <= cfg.points; i++) {
            var t = i / cfg.points;
            var theta = t * total;
            var r = cfg.a + cfg.b * theta;

            if (cfg.tremor > 0) {
                /* Tremor grows slightly with distance from centre, as a real
                   hand's does once the arm extends. */
                var growth = 0.35 + 0.65 * t;
                var oscillation = Math.sin(theta * cfg.tremorHz) * 0.6;
                var wander = noise(t) * 0.8;
                r += (oscillation + wander) * cfg.tremor * growth;
            }

            var x = cfg.cx + r * Math.cos(theta);
            var y = cfg.cy + r * Math.sin(theta);
            d += (i === 0 ? "M" : "L") + x.toFixed(2) + " " + y.toFixed(2);
        }
        return d;
    }

    /** Build a standalone <svg> element containing one spiral. */
    function spiralSvg(o) {
        var cfg = Object.assign({ size: 200, stroke: 2, label: null }, o || {});
        var ns = "http://www.w3.org/2000/svg";
        var svg = document.createElementNS(ns, "svg");

        svg.setAttribute("viewBox", "0 0 " + cfg.size + " " + cfg.size);
        svg.setAttribute("fill", "none");
        if (cfg.label) {
            svg.setAttribute("role", "img");
            svg.setAttribute("aria-label", cfg.label);
        } else {
            svg.setAttribute("aria-hidden", "true");
        }

        var path = document.createElementNS(ns, "path");
        path.setAttribute("d", spiralPath(Object.assign({
            cx: cfg.size / 2, cy: cfg.size / 2
        }, cfg)));
        path.setAttribute("stroke", "currentColor");
        path.setAttribute("stroke-width", cfg.stroke);
        path.setAttribute("stroke-linecap", "round");
        svg.appendChild(path);

        return svg;
    }

    /**
     * Animate a path drawing itself in. Honours prefers-reduced-motion by
     * rendering the finished state immediately.
     */
    function drawIn(path, duration, delay) {
        var length = path.getTotalLength();
        path.style.strokeDasharray = length;

        if (reduceMotion.matches) {
            path.style.strokeDashoffset = 0;
            return;
        }

        path.style.strokeDashoffset = length;
        path.style.transition = "stroke-dashoffset " + (duration || 1600) +
            "ms cubic-bezier(0.22, 0.61, 0.36, 1) " + (delay || 0) + "ms";
        /* Force layout so the transition has a start value to animate from. */
        void path.getBoundingClientRect();
        requestAnimationFrame(function () { path.style.strokeDashoffset = 0; });
    }

    /** Draw spirals in only once they scroll into view. */
    function drawWhenVisible(el, fn) {
        if (!("IntersectionObserver" in global)) { fn(); return; }
        var io = new IntersectionObserver(function (entries) {
            entries.forEach(function (entry) {
                if (entry.isIntersecting) { fn(); io.unobserve(entry.target); }
            });
        }, { threshold: 0.25 });
        io.observe(el);
    }

    /* ---- Section divider: a single spiral arc, flattened ---- */
    function dividerSvg() {
        var ns = "http://www.w3.org/2000/svg";
        var svg = document.createElementNS(ns, "svg");
        svg.setAttribute("viewBox", "0 0 320 46");
        svg.setAttribute("fill", "none");
        svg.setAttribute("aria-hidden", "true");

        var g = document.createElementNS(ns, "g");
        g.setAttribute("transform", "translate(160 23) scale(1 0.42)");

        var path = document.createElementNS(ns, "path");
        path.setAttribute("d", spiralPath({
            cx: 0, cy: 0, turns: 2.1, a: 1, b: 7.5, points: 400
        }));
        path.setAttribute("stroke", "currentColor");
        path.setAttribute("stroke-width", "1.25");
        g.appendChild(path);
        svg.appendChild(g);

        /* Flanking rules so the arc reads as a divider, not an ornament. */
        [[0, 56], [264, 320]].forEach(function (seg) {
            var line = document.createElementNS(ns, "line");
            line.setAttribute("x1", seg[0]); line.setAttribute("y1", 23);
            line.setAttribute("x2", seg[1]); line.setAttribute("y2", 23);
            line.setAttribute("stroke", "currentColor");
            line.setAttribute("stroke-width", "1");
            svg.appendChild(line);
        });
        return svg;
    }

    /* ---- Background texture: faint guilloche, written to a CSS variable ---- */
    function installTexture() {
        var d = spiralPath({
            cx: 550, cy: 550, turns: 26, a: 6, b: 20, points: 5200
        });
        var svg = '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="1100" viewBox="0 0 1100 1100">' +
            '<path d="' + d + '" fill="none" stroke="%2316191a" stroke-width="1.1" opacity="0.045"/></svg>';
        document.documentElement.style.setProperty(
            "--spiral-texture", "url('data:image/svg+xml;utf8," + svg + "')"
        );
    }

    global.NeuroSpiral = {
        path: spiralPath,
        svg: spiralSvg,
        drawIn: drawIn,
        drawWhenVisible: drawWhenVisible,
        divider: dividerSvg,
        installTexture: installTexture,
        reduceMotion: reduceMotion
    };
})(window);
