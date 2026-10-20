/* ==========================================================================
   NeuroSense — interface behaviour
   Progressive enhancement only: every form works with JS disabled.
   ========================================================================== */

(function () {
    "use strict";

    var S = window.NeuroSpiral;

    /* ---- Background texture + logo marks + dividers ---- */
    function installIdentity() {
        S.installTexture();

        document.querySelectorAll("[data-spiral-mark]").forEach(function (slot) {
            var svg = S.svg({
                size: 100, turns: 3.4, a: 3, b: 4.2, stroke: 5,
                points: 480, label: slot.getAttribute("data-label") || null
            });
            slot.appendChild(svg);
        });

        document.querySelectorAll("[data-spiral-divider]").forEach(function (slot) {
            slot.appendChild(S.divider());
        });
    }

    /* ---- Hero: steady vs tremor, drawn in on view ---- */
    function installHero() {
        document.querySelectorAll("[data-spiral-figure]").forEach(function (slot) {
            var tremor = parseFloat(slot.getAttribute("data-tremor")) || 0;
            var seed = parseInt(slot.getAttribute("data-seed"), 10) || 7;
            var svg = S.svg({
                size: 240, turns: 4.6, a: 2, b: 3.6, stroke: 2.25,
                points: 1100, tremor: tremor, seed: seed
            });
            slot.appendChild(svg);

            var path = svg.querySelector("path");
            S.drawWhenVisible(slot, function () {
                S.drawIn(path, 1900, tremor > 0 ? 260 : 0);
            });
        });
    }

    /* ---- File pickers: confirm the selection in text, not colour alone ---- */
    function installFilePickers() {
        document.querySelectorAll(".filepicker input[type=file]").forEach(function (input) {
            var status = input.parentElement.querySelector(".filepicker__status");
            if (!status) return;

            input.addEventListener("change", function () {
                if (!input.files || !input.files.length) {
                    status.textContent = "";
                    return;
                }
                var f = input.files[0];
                var kb = f.size / 1024;
                var size = kb > 1024 ? (kb / 1024).toFixed(1) + " MB" : Math.round(kb) + " KB";
                status.textContent = "Selected: " + f.name + "  (" + size + ")";
                input.dispatchEvent(new CustomEvent("neurosense:file", {
                    bubbles: true, detail: { file: f }
                }));
            });
        });
    }

    /* ---- Submit state: spinning spiral, and the button says what it's doing ---- */
    function installSubmitState() {
        document.querySelectorAll("form[data-analysis-form]").forEach(function (form) {
            form.addEventListener("submit", function () {
                var btn = form.querySelector("button[type=submit]");
                if (!btn) return;
                btn.disabled = true;
                btn.setAttribute("aria-busy", "true");
                btn.innerHTML =
                    '<span class="spinner" aria-hidden="true"></span> Analysing…';

                var live = document.getElementById("live-region");
                if (live) live.textContent = "Analysing. This can take a few seconds.";
            });
        });
    }

    /* ---- Confidence bars ----------------------------------------------
       The true width is rendered server-side in the style attribute, so a bar
       is correct with JavaScript disabled, blocked, or still loading. The
       growth animation therefore works in reverse: collapse to zero, then
       release back to the rendered value. Anything that goes wrong leaves the
       bar showing the truth rather than an empty track.

       A bar must never disagree with the number printed beside it. */
    function installBars() {
        if (S.reduceMotion.matches) return;   // already correct; leave it alone

        document.querySelectorAll("[data-bar-value]").forEach(function (bar) {
            var fill = bar.querySelector(".bar__fill");
            if (!fill) return;

            var target = fill.style.width;
            if (!target) return;              // nothing rendered: do not touch

            var released = false;
            function release() {
                if (released) return;
                released = true;
                fill.style.width = target;
            }

            fill.style.width = "0%";
            S.drawWhenVisible(bar, function () {
                requestAnimationFrame(release);
            });

            /* Guarantee the value is restored even if the observer never
               reports the bar as visible (fast scroll, anchor jump, find). */
            setTimeout(release, 1200);
        });
    }

    /* ---- Results: ease in, and move focus there after a submission ---- */
    function installResults() {
        var results = document.getElementById("results");
        if (!results) return;

        results.classList.add("is-revealed");

        var heading = results.querySelector("h2, h3");
        if (heading) {
            heading.setAttribute("tabindex", "-1");
            heading.focus({ preventScroll: true });
            heading.scrollIntoView({
                behavior: S.reduceMotion.matches ? "auto" : "smooth",
                block: "start"
            });
        }
    }

    document.addEventListener("DOMContentLoaded", function () {
        installIdentity();
        installHero();
        installFilePickers();
        installSubmitState();
        installBars();
        installResults();
    });
})();
