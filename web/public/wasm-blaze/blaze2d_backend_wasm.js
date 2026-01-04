let wasm;

let cachedUint8ArrayMemory0 = null;

function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });

cachedTextDecoder.decode();

const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let heap = new Array(128).fill(undefined);

heap.push(undefined, null, true, false);

function getObject(idx) { return heap[idx]; }

let heap_next = heap.length;

function addHeapObject(obj) {
    if (heap_next === heap.length) heap.push(heap.length + 1);
    const idx = heap_next;
    heap_next = heap[idx];

    heap[idx] = obj;
    return idx;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        wasm.__wbindgen_export(addHeapObject(e));
    }
}

let WASM_VECTOR_LEN = 0;

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    }
}

function passStringToWasm0(arg, malloc, realloc) {

    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }

    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

let cachedDataViewMemory0 = null;

function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

function dropObject(idx) {
    if (idx < 132) return;
    heap[idx] = heap_next;
    heap_next = idx;
}

function takeObject(idx) {
    const ret = getObject(idx);
    dropObject(idx);
    return ret;
}

let cachedUint32ArrayMemory0 = null;

function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

function passArray32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getUint32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

function getArrayU32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}
/**
 * Check if streaming mode is supported.
 * @returns {boolean}
 */
export function isStreamingSupported() {
    const ret = wasm.isSelectiveSupported();
    return ret !== 0;
}

/**
 * Get the library version.
 * @returns {string}
 */
export function getVersion() {
    let deferred1_0;
    let deferred1_1;
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        wasm.getVersion(retptr);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        deferred1_0 = r0;
        deferred1_1 = r1;
        return getStringFromWasm0(r0, r1);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
        wasm.__wbindgen_export2(deferred1_0, deferred1_1, 1);
    }
}

/**
 * Get supported solver types.
 * @returns {Array<any>}
 */
export function getSupportedSolvers() {
    const ret = wasm.getSupportedSolvers();
    return takeObject(ret);
}

/**
 * Initialize the WASM module with proper panic handling.
 * Call this once at the start of your application.
 *
 * This sets up the panic hook so that Rust panics are printed
 * to the browser console with full stack traces instead of
 * just showing "RuntimeError: unreachable".
 */
export function initPanicHook() {
    wasm.initPanicHook();
}

/**
 * Check if selective filtering is supported.
 * @returns {boolean}
 */
export function isSelectiveSupported() {
    const ret = wasm.isSelectiveSupported();
    return ret !== 0;
}

const WasmBackendFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmbackend_free(ptr >>> 0, 1));

export class WasmBackend {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmBackendFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmbackend_free(ptr, 0);
    }
}
if (Symbol.dispose) WasmBackend.prototype[Symbol.dispose] = WasmBackend.prototype.free;

const WasmBulkDriverFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmbulkdriver_free(ptr >>> 0, 1));
/**
 * WebAssembly wrapper for the bulk driver with streaming support.
 *
 * This provides a unified interface for running band structure calculations
 * in the browser, with support for both Maxwell and EA solvers.
 */
export class WasmBulkDriver {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmBulkDriverFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmbulkdriver_free(ptr, 0);
    }
    /**
     * Check if this is a Maxwell solver.
     * @returns {boolean}
     */
    get isMaxwell() {
        const ret = wasm.wasmbulkdriver_isMaxwell(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Run and return all results as an array (COLLECT mode).
     *
     * @returns Object with `results` (array) and `stats`
     * @returns {any}
     */
    runCollect() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.wasmbulkdriver_runCollect(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return takeObject(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Get the solver type ("maxwell" or "ea").
     * @returns {string}
     */
    get solverType() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.wasmbulkdriver_solverType(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export2(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Get the first N expanded job configurations.
     * @param {number} n
     * @returns {Array<any>}
     */
    getJobConfigs(n) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.wasmbulkdriver_getJobConfigs(retptr, this.__wbg_ptr, n);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return takeObject(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Run the computation with a callback for each result (STREAM mode).
     *
     * @param callback - Function(result) called for each completed job
     * @returns Statistics object
     * @param {Function} callback
     * @returns {any}
     */
    runWithCallback(callback) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.wasmbulkdriver_runWithCallback(retptr, this.__wbg_ptr, addHeapObject(callback));
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return takeObject(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Run and collect with optional filtering.
     * @param {Uint32Array | null} [k_indices]
     * @param {Uint32Array | null} [band_indices]
     * @returns {any}
     */
    runCollectFiltered(k_indices, band_indices) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            var ptr0 = isLikeNone(k_indices) ? 0 : passArray32ToWasm0(k_indices, wasm.__wbindgen_export3);
            var len0 = WASM_VECTOR_LEN;
            var ptr1 = isLikeNone(band_indices) ? 0 : passArray32ToWasm0(band_indices, wasm.__wbindgen_export3);
            var len1 = WASM_VECTOR_LEN;
            wasm.wasmbulkdriver_runCollectFiltered(retptr, this.__wbg_ptr, ptr0, len0, ptr1, len1);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return takeObject(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Run with streaming output and server-side filtering (SELECTIVE mode).
     *
     * @param k_indices - Array of k-point indices to include (0-based), or null for all
     * @param band_indices - Array of band indices to include (0-based), or null for all
     * @param callback - Function(result) called for each filtered result
     * @returns Statistics object
     * @param {Uint32Array | null | undefined} k_indices
     * @param {Uint32Array | null | undefined} band_indices
     * @param {Function} callback
     * @returns {any}
     */
    runStreamingFiltered(k_indices, band_indices, callback) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            var ptr0 = isLikeNone(k_indices) ? 0 : passArray32ToWasm0(k_indices, wasm.__wbindgen_export3);
            var len0 = WASM_VECTOR_LEN;
            var ptr1 = isLikeNone(band_indices) ? 0 : passArray32ToWasm0(band_indices, wasm.__wbindgen_export3);
            var len1 = WASM_VECTOR_LEN;
            wasm.wasmbulkdriver_runStreamingFiltered(retptr, this.__wbg_ptr, ptr0, len0, ptr1, len1, addHeapObject(callback));
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return takeObject(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Run with K-POINT STREAMING: callback is called after EACH k-point is solved.
     *
     * This is the preferred mode for real-time visualization of band structure
     * computation. The callback receives incremental results as each k-point
     * completes, enabling smooth progressive rendering of the band diagram.
     *
     * @param callback - Function(kPointResult) called after each k-point solve
     * @returns Statistics object with completion info
     *
     * The callback receives an object with:
     * - `stream_type`: "k_point" (to distinguish from job-level streaming)
     * - `k_index`: Index of this k-point (0-based)
     * - `total_k_points`: Total number of k-points
     * - `k_point`: [kx, ky] in fractional coordinates
     * - `distance`: Cumulative path distance to this k-point
     * - `omegas`: Array of frequencies for all bands at this k-point
     * - `bands`: Same as omegas (alias for convenience)
     * - `iterations`: Number of LOBPCG iterations for this k-point
     * - `is_gamma`: Whether this is a Γ-point
     * - `progress`: Completion fraction (0.0 to 1.0)
     * - `params`: Job parameters (eps_bg, resolution, polarization, etc.)
     * @param {Function} callback
     * @returns {any}
     */
    runWithKPointStreaming(callback) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.wasmbulkdriver_runWithKPointStreaming(retptr, this.__wbg_ptr, addHeapObject(callback));
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return takeObject(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Create a new bulk driver from TOML configuration.
     *
     * @param config_str - Configuration as TOML string
     * @throws Error if configuration is invalid
     * @param {string} config_str
     */
    constructor(config_str) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passStringToWasm0(config_str, wasm.__wbindgen_export3, wasm.__wbindgen_export4);
            const len0 = WASM_VECTOR_LEN;
            wasm.wasmbulkdriver_new(retptr, ptr0, len0);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            this.__wbg_ptr = r0 >>> 0;
            WasmBulkDriverFinalization.register(this, this.__wbg_ptr, this);
            return this;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Check if this is an EA solver.
     * @returns {boolean}
     */
    get isEA() {
        const ret = wasm.wasmbulkdriver_isEA(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Dry run: get job count and parameter info without executing.
     * @returns {any}
     */
    dryRun() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.wasmbulkdriver_dryRun(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return takeObject(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Get the grid dimensions as [nx, ny].
     * @returns {Array<any>}
     */
    get gridSize() {
        const ret = wasm.wasmbulkdriver_gridSize(this.__wbg_ptr);
        return takeObject(ret);
    }
    /**
     * Get the number of jobs that will be executed.
     * @returns {number}
     */
    get jobCount() {
        const ret = wasm.wasmbulkdriver_jobCount(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Get the number of bands being computed.
     * @returns {number}
     */
    get numBands() {
        const ret = wasm.wasmbulkdriver_numBands(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) WasmBulkDriver.prototype[Symbol.dispose] = WasmBulkDriver.prototype.free;

const WasmSelectiveFilterFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmselectivefilter_free(ptr >>> 0, 1));
/**
 * WASM wrapper for selective filtering configuration.
 */
export class WasmSelectiveFilter {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmSelectiveFilterFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmselectivefilter_free(ptr, 0);
    }
    /**
     * Get the number of bands in the filter.
     * @returns {number | undefined}
     */
    get bandCount() {
        const ret = wasm.wasmselectivefilter_bandCount(this.__wbg_ptr);
        return ret === 0x100000001 ? undefined : ret;
    }
    /**
     * Get band indices as array.
     * @returns {Uint32Array}
     */
    get bandIndices() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.wasmselectivefilter_bandIndices(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v1 = getArrayU32FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export2(r0, r1 * 4, 4);
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Set k-point indices to include.
     * @param {Uint32Array} indices
     */
    setKIndices(indices) {
        const ptr0 = passArray32ToWasm0(indices, wasm.__wbindgen_export3);
        const len0 = WASM_VECTOR_LEN;
        wasm.wasmselectivefilter_setKIndices(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Set band indices to include.
     * @param {Uint32Array} indices
     */
    setBandIndices(indices) {
        const ptr0 = passArray32ToWasm0(indices, wasm.__wbindgen_export3);
        const len0 = WASM_VECTOR_LEN;
        wasm.wasmselectivefilter_setBandIndices(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Create a new selective filter (pass-through by default).
     */
    constructor() {
        const ret = wasm.wasmselectivefilter_new();
        this.__wbg_ptr = ret >>> 0;
        WasmSelectiveFilterFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Clear all filter settings.
     */
    clear() {
        wasm.wasmselectivefilter_clear(this.__wbg_ptr);
    }
    /**
     * Get the number of k-points in the filter.
     * @returns {number | undefined}
     */
    get kCount() {
        const ret = wasm.wasmselectivefilter_kCount(this.__wbg_ptr);
        return ret === 0x100000001 ? undefined : ret;
    }
    /**
     * Check if the filter is active.
     * @returns {boolean}
     */
    get isActive() {
        const ret = wasm.wasmselectivefilter_isActive(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Get k-indices as array.
     * @returns {Uint32Array}
     */
    get kIndices() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.wasmselectivefilter_kIndices(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v1 = getArrayU32FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export2(r0, r1 * 4, 4);
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
}
if (Symbol.dispose) WasmSelectiveFilter.prototype[Symbol.dispose] = WasmSelectiveFilter.prototype.free;

const EXPECTED_RESPONSE_TYPES = new Set(['basic', 'cors', 'default']);

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);

            } catch (e) {
                const validResponse = module.ok && EXPECTED_RESPONSE_TYPES.has(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else {
                    throw e;
                }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);

    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };

        } else {
            return instance;
        }
    }
}

function __wbg_get_imports() {
    const imports = {};
    imports.wbg = {};
    imports.wbg.__wbg___wbindgen_throw_b855445ff6a94295 = function(arg0, arg1) {
        throw new Error(getStringFromWasm0(arg0, arg1));
    };
    imports.wbg.__wbg_call_525440f72fbfc0ea = function() { return handleError(function (arg0, arg1, arg2) {
        const ret = getObject(arg0).call(getObject(arg1), getObject(arg2));
        return addHeapObject(ret);
    }, arguments) };
    imports.wbg.__wbg_error_7534b8e9a36f1ab4 = function(arg0, arg1) {
        let deferred0_0;
        let deferred0_1;
        try {
            deferred0_0 = arg0;
            deferred0_1 = arg1;
            console.error(getStringFromWasm0(arg0, arg1));
        } finally {
            wasm.__wbindgen_export2(deferred0_0, deferred0_1, 1);
        }
    };
    imports.wbg.__wbg_new_1acc0b6eea89d040 = function() {
        const ret = new Object();
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_new_8a6f238a6ece86ea = function() {
        const ret = new Error();
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_new_e17d9f43105b08be = function() {
        const ret = new Array();
        return addHeapObject(ret);
    };
    imports.wbg.__wbg_now_793306c526e2e3b6 = function() {
        const ret = Date.now();
        return ret;
    };
    imports.wbg.__wbg_push_df81a39d04db858c = function(arg0, arg1) {
        const ret = getObject(arg0).push(getObject(arg1));
        return ret;
    };
    imports.wbg.__wbg_set_c2abbebe8b9ebee1 = function() { return handleError(function (arg0, arg1, arg2) {
        const ret = Reflect.set(getObject(arg0), getObject(arg1), getObject(arg2));
        return ret;
    }, arguments) };
    imports.wbg.__wbg_stack_0ed75d68575b0f3c = function(arg0, arg1) {
        const ret = getObject(arg1).stack;
        const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_export3, wasm.__wbindgen_export4);
        const len1 = WASM_VECTOR_LEN;
        getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
    };
    imports.wbg.__wbindgen_cast_2241b6af4c4b2941 = function(arg0, arg1) {
        // Cast intrinsic for `Ref(String) -> Externref`.
        const ret = getStringFromWasm0(arg0, arg1);
        return addHeapObject(ret);
    };
    imports.wbg.__wbindgen_cast_d6cd19b81560fd6e = function(arg0) {
        // Cast intrinsic for `F64 -> Externref`.
        const ret = arg0;
        return addHeapObject(ret);
    };
    imports.wbg.__wbindgen_object_drop_ref = function(arg0) {
        takeObject(arg0);
    };

    return imports;
}

function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    __wbg_init.__wbindgen_wasm_module = module;
    cachedDataViewMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;



    return wasm;
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (typeof module !== 'undefined') {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();

    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }

    const instance = new WebAssembly.Instance(module, imports);

    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (typeof module_or_path !== 'undefined') {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (typeof module_or_path === 'undefined') {
        module_or_path = new URL('blaze2d_backend_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync };
export default __wbg_init;
