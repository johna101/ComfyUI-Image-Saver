import { app } from '../../scripts/app.js'

// Component-row editor for the Workflow Metadata Resolver node.
//
// The node is a pure declaration: each row binds a metadata `field` to a value in
// the graph (`#node_id.input`). Rows are edited as components (label + node picker
// + input picker), not free text — the underlying `bindings` string widget is kept
// (hidden) purely so the rows serialize into the workflow/prompt, where the gallery
// reads them back. Node IDs are stored; node *names* are displayed.

const RESOLVER_NODE = "Workflow Metadata Resolver (Image Saver)";
const BINDINGS_WIDGET = "bindings";

// Suggested field names for common inputs, so captured rows read cleanly.
const FIELD_ALIASES = {
    ckpt_name: "model", unet_name: "model", model_name: "model",
    sampler_name: "sampler", scheduler: "scheduler", noise_seed: "seed",
};

// ---- binding string <-> rows ------------------------------------------------

function separatorIndex(line) {
    const found = [":", "="].map(c => line.indexOf(c)).filter(i => i >= 0);
    return found.length ? Math.min(...found) : -1;
}

/** Parse the bindings string into rows [{field, nodeId|null, input|null}]. */
function parseRows(text) {
    const rows = [];
    for (const raw of String(text ?? "").split("\n")) {
        const line = raw.trim();
        if (!line || line.startsWith("//")) continue;
        const idx = separatorIndex(line);
        if (idx < 0) continue;
        const field = line.slice(0, idx).trim();
        if (!field) continue;
        const pointer = line.slice(idx + 1).trim().replace(/^#/, "").trim();
        if (!pointer) { rows.push({ field, nodeId: null, input: null }); continue; }
        const dot = pointer.indexOf(".");
        if (dot < 0) { rows.push({ field, nodeId: null, input: null }); continue; }
        rows.push({ field, nodeId: pointer.slice(0, dot).trim(), input: pointer.slice(dot + 1).trim() });
    }
    return rows;
}

/** Serialize rows back to the bindings string (`field: #id.input`, or `field:` if unbound). */
function serializeRows(rows) {
    return rows
        .filter(r => r.field)
        .map(r => (r.nodeId && r.input) ? `${r.field}: #${r.nodeId}.${r.input}` : `${r.field}:`)
        .join("\n");
}

// ---- graph helpers ----------------------------------------------------------

function nodeClass(node) { return node?.type ?? node?.comfyClass ?? ""; }
function nodeLabel(node) { return `${node.title || nodeClass(node)} (#${node.id})`; }

/** Union of a node's widget names and input names. */
function fieldNames(node) {
    const names = new Set();
    for (const w of node?.widgets ?? []) if (w?.name) names.add(w.name);
    for (const i of node?.inputs ?? []) if (i?.name) names.add(i.name);
    return names;
}

/** Capturable fields of a node: widgets, plus linked inputs (resolvable via wire-follow). */
function captureFields(node) {
    const seen = new Set();
    const out = [];
    for (const w of node?.widgets ?? []) {
        if (!w?.name || w.type === "button" || seen.has(w.name)) continue;
        seen.add(w.name); out.push(w.name);
    }
    for (const i of node?.inputs ?? []) {
        if (!i?.name || i.link == null || seen.has(i.name)) continue;
        seen.add(i.name); out.push(i.name);
    }
    return out;
}

function inputSource(graph, node, inputName) {
    const input = (node?.inputs ?? []).find(i => i?.name === inputName);
    if (!input || input.link == null) return null;
    const link = graph.links?.[input.link];
    return link ? (graph.getNodeById(link.origin_id) ?? null) : null;
}

function findSampler(graph) {
    return (graph?._nodes ?? []).find(n => /KSampler|SamplerCustom/.test(nodeClass(n))) ?? null;
}

function traceToWidget(graph, node, inputName, targets, depth = 0) {
    if (depth > 16) return null;
    const src = inputSource(graph, node, inputName);
    if (!src) return null;
    const names = fieldNames(src);
    for (const t of targets) if (names.has(t)) return { node: src, widget: t };
    if (names.has(inputName)) return traceToWidget(graph, src, inputName, targets, depth + 1);
    return null;
}

function traceToSize(graph, node, inputName, depth = 0) {
    if (depth > 16) return null;
    const src = inputSource(graph, node, inputName);
    if (!src) return null;
    const names = fieldNames(src);
    if (names.has("width") && names.has("height")) return src;
    for (const p of ["latent_image", "samples", "latent"]) {
        if (names.has(p)) { const r = traceToSize(graph, src, p, depth + 1); if (r) return r; }
    }
    return null;
}

/** Power Lora Loader (rgthree) lora slots: widgets whose value is a {lora, on,...} dict. */
function loraSlots(node) {
    const slots = [];
    for (const w of node?.widgets ?? []) {
        if (w?.name && w.value && typeof w.value === "object" && typeof w.value.lora === "string" && w.value.lora) {
            slots.push({ input: w.name, enabled: w.value.on !== false });
        }
    }
    return slots;
}

/** Trace a sampler-centred graph into rows [{field, nodeId, input}] (mirrors the gallery parser). */
function autoFillRows(graph) {
    const sampler = findSampler(graph);
    if (!sampler) return [];

    const sid = String(sampler.id);
    const names = fieldNames(sampler);
    const out = [];
    const add = (field, nodeId, input) => out.push({ field, nodeId: String(nodeId), input });

    for (const [field, input] of [["steps", "steps"], ["cfg", "cfg"], ["sampler", "sampler_name"], ["scheduler", "scheduler"], ["denoise", "denoise"]]) {
        if (names.has(input)) add(field, sid, input);
    }
    if (names.has("seed")) add("seed", sid, "seed");
    else if (names.has("noise_seed")) add("seed", sid, "noise_seed");
    if (names.has("positive")) add("positive", sid, "positive");
    if (names.has("negative")) add("negative", sid, "negative");

    const loader = traceToWidget(graph, sampler, "model", ["ckpt_name", "unet_name"]);
    if (loader) add("model", loader.node.id, loader.widget);

    const latent = traceToSize(graph, sampler, "latent_image");
    if (latent) { add("width", latent.id, "width"); add("height", latent.id, "height"); }

    // LoRAs: one row per enabled slot across any Power Lora Loader in the graph.
    for (const n of graph?._nodes ?? []) {
        if (!/Power Lora Loader/i.test(nodeClass(n))) continue;
        for (const slot of loraSlots(n)) {
            if (slot.enabled) add(slot.input, n.id, slot.input);
        }
    }
    return out;
}

// ---- row editing ------------------------------------------------------------

function syncRows(node) {
    // Row state is the source of truth; the bindings DOM widget reads it on
    // demand via getValue (see setupResolverEditor), so a redraw is all we need.
    node.graph?.setDirtyCanvas(true, true);
}

/** Remove stale output slots — the node has no outputs, but workflows saved with
 *  the older definition restore them on load. */
function stripOutputs(node) {
    if (!node.outputs) return;
    while (node.outputs.length) node.removeOutput(node.outputs.length - 1);
}

/** Insert or update a row for `field`, pointing at #nodeId.input. Returns the row. */
function upsertRow(node, field, nodeId, input) {
    const rows = node._resolverRows;
    let row = rows.find(r => r.field === field);
    if (!row) { row = { field, nodeId: null, input: null }; rows.push(row); }
    row.nodeId = nodeId != null ? String(nodeId) : null;
    row.input = input ?? null;
    return row;
}

function suggestFieldName(node, inputName) {
    if (inputName === "text") {
        const bound = new Set((node._resolverRows ?? []).map(r => r.field));
        if (!bound.has("positive")) return "positive";
        if (!bound.has("negative")) return "negative";
        return "prompt";
    }
    return FIELD_ALIASES[inputName] ?? inputName;
}

// ---- DOM widget -------------------------------------------------------------

function injectStyles() {
    if (document.getElementById("imgsaver-resolver-css")) return;
    const style = document.createElement("style");
    style.id = "imgsaver-resolver-css";
    style.textContent = `
        .imgsaver-resolver { display:flex; flex-direction:column; gap:3px; padding:4px 2px; font-size:11px; box-sizing:border-box; }
        .imgsaver-resolver .row { display:flex; gap:3px; align-items:center; }
        .imgsaver-resolver input.field { flex:0 0 84px; min-width:0; }
        .imgsaver-resolver select { flex:1 1 0; min-width:0; }
        .imgsaver-resolver input, .imgsaver-resolver select {
            background:#222; color:#ddd; border:1px solid #444; border-radius:4px; padding:2px 4px; font-size:11px; height:20px; }
        .imgsaver-resolver .row.invalid input.field { border-color:#c0504d; }
        .imgsaver-resolver .row.invalid select.bad { border-color:#c0504d; color:#e08; }
        .imgsaver-resolver button { background:#333; color:#ddd; border:1px solid #444; border-radius:4px; cursor:pointer; font-size:11px; height:20px; }
        .imgsaver-resolver button.remove { flex:0 0 22px; }
        .imgsaver-resolver .toolbar { display:flex; gap:4px; margin-top:2px; }
        .imgsaver-resolver .toolbar button { flex:1 1 0; height:22px; }
    `;
    document.head.appendChild(style);
}

function makeNodeSelect(graph, row) {
    const sel = document.createElement("select");
    const blank = new Option("(unbound)", "");
    sel.add(blank);
    const nodes = (graph?._nodes ?? []).filter(n => nodeClass(n) !== RESOLVER_NODE)
        .sort((a, b) => nodeLabel(a).localeCompare(nodeLabel(b)));
    for (const n of nodes) sel.add(new Option(nodeLabel(n), String(n.id)));

    if (row.nodeId != null) {
        const exists = (graph?._nodes ?? []).some(n => String(n.id) === row.nodeId);
        if (!exists) sel.add(new Option(`#${row.nodeId} (missing)`, row.nodeId));
        sel.value = row.nodeId;
    } else {
        sel.value = "";
    }
    return sel;
}

function fillInputSelect(sel, graph, row) {
    sel.innerHTML = "";
    sel.add(new Option("(field)", ""));
    const node = row.nodeId != null ? graph?.getNodeById?.(Number(row.nodeId)) : null;
    const fields = node ? captureFields(node) : [];
    for (const f of fields) sel.add(new Option(f, f));
    if (row.input && !fields.includes(row.input)) sel.add(new Option(`${row.input} (missing)`, row.input));
    sel.value = row.input ?? "";
}

function markValidity(rowEl, graph, row, nodeSel, inputSel) {
    const node = row.nodeId != null ? graph?.getNodeById?.(Number(row.nodeId)) : null;
    const nodeMissing = row.nodeId != null && !node;
    const inputMissing = !!row.input && (!node || !fieldNames(node).has(row.input));
    rowEl.classList.toggle("invalid", nodeMissing || inputMissing);
    nodeSel.classList.toggle("bad", nodeMissing);
    inputSel.classList.toggle("bad", inputMissing);
    rowEl.title = nodeMissing ? `Node #${row.nodeId} is not in this workflow`
        : inputMissing ? `Input '${row.input}' is not on that node` : "";
}

function renderRows(node) {
    const graph = node.graph;
    const list = node._resolverListEl;
    if (!list) return;
    list.innerHTML = "";

    node._resolverRows.forEach((row, i) => {
        const rowEl = document.createElement("div");
        rowEl.className = "row";

        const field = document.createElement("input");
        field.className = "field"; field.type = "text"; field.value = row.field;
        field.placeholder = "field"; field.title = "Metadata field name";

        const nodeSel = makeNodeSelect(graph, row);
        const inputSel = document.createElement("select");
        fillInputSelect(inputSel, graph, row);

        const remove = document.createElement("button");
        remove.className = "remove"; remove.textContent = "✕"; remove.title = "Remove row";

        const revalidate = () => markValidity(rowEl, graph, row, nodeSel, inputSel);

        field.addEventListener("change", () => { row.field = field.value.trim(); syncRows(node); });
        nodeSel.addEventListener("change", () => {
            row.nodeId = nodeSel.value || null;
            row.input = null;
            fillInputSelect(inputSel, graph, row);
            revalidate(); syncRows(node);
        });
        inputSel.addEventListener("change", () => { row.input = inputSel.value || null; revalidate(); syncRows(node); });
        remove.addEventListener("click", () => { node._resolverRows.splice(i, 1); renderRows(node); syncRows(node); });

        rowEl.append(field, nodeSel, inputSel, remove);
        list.appendChild(rowEl);
        revalidate();
    });

    node._resolverResize?.();
}

function setupResolverEditor(node) {
    if (node._resolverListEl) return;

    // Take the auto-created text widget's value, then remove it entirely: the rows
    // are the data, and the DOM widget below serializes them as `bindings` itself
    // (no separate text box to hide).
    const idx = (node.widgets ?? []).findIndex(w => w.name === BINDINGS_WIDGET);
    const initial = idx >= 0 ? node.widgets[idx].value : "";
    if (idx >= 0) node.widgets.splice(idx, 1);

    injectStyles();
    node._resolverRows = parseRows(initial);

    const container = document.createElement("div");
    container.className = "imgsaver-resolver";
    const list = document.createElement("div");
    node._resolverListEl = list;
    container.appendChild(list);

    const toolbar = document.createElement("div");
    toolbar.className = "toolbar";
    const addBtn = document.createElement("button");
    addBtn.textContent = "+ Add"; addBtn.title = "Add a binding row";
    const autoBtn = document.createElement("button");
    autoBtn.textContent = "Auto-fill"; autoBtn.title = "Trace the sampler and fill common fields + LoRAs";
    toolbar.append(addBtn, autoBtn);
    container.appendChild(toolbar);

    addBtn.addEventListener("click", () => {
        node._resolverRows.push({ field: "", nodeId: null, input: null });
        renderRows(node); syncRows(node);
        list.lastChild?.querySelector("input.field")?.focus();
    });
    autoBtn.addEventListener("click", () => {
        for (const b of autoFillRows(node.graph)) upsertRow(node, b.field, b.nodeId, b.input);
        renderRows(node); syncRows(node);
    });

    // The DOM widget IS the `bindings` input: it serializes the rows into the
    // prompt (which the gallery reads), so there is no separate text widget.
    const domWidget = node.addDOMWidget(BINDINGS_WIDGET, "resolver_rows", container, {
        serialize: true,
        getValue: () => serializeRows(node._resolverRows),
        setValue: (v) => { node._resolverRows = parseRows(v); renderRows(node); },
    });
    node._resolverResize = () => {
        const rows = node._resolverRows.length;
        domWidget.computeSize = (w) => [w, 26 * rows + 34];
        node.setSize(node.computeSize());
        node.graph?.setDirtyCanvas(true, true);
    };

    renderRows(node);
}

// ---- extension registration -------------------------------------------------

app.registerExtension({
    name: "ComfyUI-Image-Saver.MetadataResolverPicker",

    beforeRegisterNodeDef(nodeType, nodeData) {
        // "Send to Metadata Resolver" on every other node's right-click menu.
        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
            getExtraMenuOptions?.apply(this, arguments);
            if (nodeClass(this) === RESOLVER_NODE) return;
            const fields = captureFields(this);
            if (!fields.length) return;

            const source = this;
            options.push({
                content: "Send to Metadata Resolver",
                has_submenu: true,
                submenu: {
                    options: fields.map(input => ({
                        content: input,
                        callback: (v, o, e) => sendCapture(source, input, e),
                    })),
                },
            });
        };

        if (nodeData?.name === RESOLVER_NODE) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                setupResolverEditor(this);
                stripOutputs(this);
            };
            // On load: drop the stale output slots and rebuild rows from the saved
            // bindings string (read from widgets_values, robust to whether the host
            // routed it through the DOM widget's setValue).
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                onConfigure?.apply(this, arguments);
                stripOutputs(this);
                if (!this._resolverListEl) return;
                const vals = info?.widgets_values;
                let saved = null;
                if (Array.isArray(vals)) saved = vals.find(v => typeof v === "string");
                else if (vals && typeof vals === "object") saved = vals[BINDINGS_WIDGET];
                if (typeof saved === "string") {
                    this._resolverRows = parseRows(saved);
                    renderRows(this);
                }
            };
        }
    },
});

function sendCapture(sourceNode, input, event) {
    const graph = sourceNode.graph;
    const resolvers = (graph?._nodes ?? []).filter(n => nodeClass(n) === RESOLVER_NODE);

    const bind = (resolver) => {
        const field = suggestFieldName(resolver, input);
        upsertRow(resolver, field, sourceNode.id, input);
        renderRows(resolver); syncRows(resolver);
    };

    if (resolvers.length === 0) {
        const resolver = LiteGraph.createNode(RESOLVER_NODE);
        if (!resolver) return;
        graph.add(resolver);
        resolver.pos = [sourceNode.pos[0] + (sourceNode.size?.[0] ?? 200) + 40, sourceNode.pos[1]];
        bind(resolver);
    } else if (resolvers.length === 1) {
        bind(resolvers[0]);
    } else {
        new LiteGraph.ContextMenu(
            resolvers.map(r => ({ content: nodeLabel(r), callback: () => bind(r) })),
            { event, title: "Choose Metadata Resolver" }
        );
    }
}
