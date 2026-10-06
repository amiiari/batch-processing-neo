// Right-click a thumbnail in either batch tab's source gallery -> select that
// image and fill its prompt box with the image's entire prompt (or, if it has
// none baked in, the live prompt of the tab it came from):
//   Batch ADetailer  -> Slot 1's ADetailer prompt, falling back to img2img's
//   Batch Hires-Fix  -> the hires prompt box, falling back to txt2img's
//
// Gradio has no contextmenu event, so this is the usual Forge dance: stash the
// clicked index (and the live prompt, for the fallback) in hidden textboxes,
// dispatch `input` so gradio's frontend picks the values up, then click a
// hidden button whose python handler does the work.
//
// Also: ←/→ steps through the thumbnails without clicking each one — it just
// clicks the neighbour of the selected one, so gradio's own select event does
// everything a real click would.
(function () {
    "use strict";

    // One entry per tab: its gallery, its hidden plumbing, and the Forge prompt
    // box to fall back to. The elem_ids are set in scripts/batch_*.py.
    const TABS = [
        {
            gallery: "batch_adetailer_source",
            index: "batch_adetailer_rclick",
            prompt: "batch_adetailer_rclick_prompt",
            button: "batch_adetailer_rclick_btn",
            livePrompt: "img2img_prompt",
        },
        {
            gallery: "batch_hires_fix_source",
            index: "batch_hires_fix_rclick",
            prompt: "batch_hires_fix_rclick_prompt",
            button: "batch_hires_fix_rclick_btn",
            livePrompt: "txt2img_prompt",
        },
    ];

    function onRightClick(tab, event) {
        const gallery = event.currentTarget;
        const thumbs = Array.from(gallery.querySelectorAll(".thumbnail-item"));
        const clicked = event.target.closest(".thumbnail-item");
        const index = thumbs.indexOf(clicked);
        if (index < 0) {
            return;  // right-click on empty gallery space: leave the browser menu alone
        }

        event.preventDefault();

        const root = gradioApp();
        const field = root.querySelector(`#${tab.index} textarea, #${tab.index} input`);
        const button = root.querySelector(`#${tab.button}`);
        if (!field || !button) {
            return;
        }

        field.value = String(index);
        field.dispatchEvent(new Event("input", { bubbles: true }));

        // Stash the live prompt so python can fall back to it when the clicked
        // image has no prompt of its own. #img2img_prompt / #txt2img_prompt live
        // on their own tabs but stay in the DOM even when those aren't showing.
        const promptField = root.querySelector(`#${tab.prompt} textarea, #${tab.prompt} input`);
        const live = root.querySelector(`#${tab.livePrompt} textarea, #${tab.livePrompt} input`);
        if (promptField) {
            promptField.value = live ? live.value : "";
            promptField.dispatchEvent(new Event("input", { bubbles: true }));
        }

        // Let gradio's input handlers commit the values before the click reads them.
        setTimeout(() => button.click(), 30);
    }

    function onArrowKey(event) {
        if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") {
            return;
        }
        if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) {
            return;
        }
        // Leave the arrows alone while typing/adjusting: prompt boxes, sliders,
        // dropdowns and the like all use them for their own cursor.
        const t = event.target;
        if (t && t.closest && t.closest("input, textarea, select, [contenteditable='true']")) {
            return;
        }

        // offsetParent picks the one whose tab is actually on screen.
        const gallery = TABS
            .map((tab) => gradioApp().querySelector(`#${tab.gallery}`))
            .find((el) => el && el.offsetParent !== null);
        if (!gallery) {
            return;  // neither batch tab is on screen
        }

        const thumbs = Array.from(gallery.querySelectorAll(".thumbnail-item"));
        if (!thumbs.length) {
            return;
        }

        const current = thumbs.findIndex((el) => el.classList.contains("selected"));
        const next = current < 0 ? 0 : current + (event.key === "ArrowRight" ? 1 : -1);
        if (next < 0 || next >= thumbs.length) {
            return;  // already at either end
        }

        event.preventDefault();
        thumbs[next].click();
        thumbs[next].scrollIntoView({ block: "nearest" });
    }

    onUiLoaded(function () {
        // The document survives a Reload UI, the gallery elements do not —
        // hence one guard per listener.
        if (!document.body.dataset.badArrowNav) {
            document.body.dataset.badArrowNav = "1";
            document.addEventListener("keydown", onArrowKey);
        }

        for (const tab of TABS) {
            const gallery = gradioApp().querySelector(`#${tab.gallery}`);
            if (!gallery || gallery.dataset.badRightClick) {
                continue;
            }
            // Delegated on the gallery itself: the thumbnails are re-rendered on
            // every drop, the container is not.
            gallery.dataset.badRightClick = "1";
            gallery.addEventListener("contextmenu", (event) => onRightClick(tab, event));
        }
    });
})();
