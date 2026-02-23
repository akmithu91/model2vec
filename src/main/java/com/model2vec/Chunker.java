package com.model2vec;

import java.nio.charset.StandardCharsets;

/**
 * Token-aware text chunker backed by a HuggingFace tokenizer.
 *
 * <h3>Output format</h3>
 * <p>{@link #chunk(String)} returns a flat {@code int[]} of UTF-8 byte-offset
 * pairs: {@code [start0, end0, start1, end1, ...]}. Reconstruct each chunk:</p>
 *
 * <pre>{@code
 * byte[] bytes   = text.getBytes(StandardCharsets.UTF_8);
 * int[]  offsets = chunker.chunk(text);
 *
 * for (int i = 0; i < offsets.length; i += 2) {
 *     int    start = offsets[i];
 *     int    end   = offsets[i + 1];
 *     String chunk = new String(bytes, start, end - start, StandardCharsets.UTF_8);
 *     // pass chunk to Model2Vec.encode(...)
 * }
 * }</pre>
 *
 * <p>Using {@code getBytes(UTF_8)} once per text and {@code new String(...)}
 * per chunk is correct for both ASCII and Unicode input. The chunker returns
 * raw UTF-8 byte offsets from the tokenizer — no conversion on the Rust side.</p>
 *
 * <h3>JNI cost</h3>
 * <p>A single {@code set_int_array_region} call (bulk memcpy) crosses the JNI
 * boundary, regardless of how many chunks were produced. There are zero
 * {@code new_string()} calls from the Rust side.</p>
 *
 * <h3>Threading</h3>
 * <p>{@link #chunk} is thread-safe. The tokenizer is read-only after construction.</p>
 *
 * <pre>{@code
 * // Default SnapCfg
 * try (Chunker c = new Chunker("bert-base-uncased", 128, 128)) {
 *     int[] offsets = c.chunk("Long article text...");
 * }
 *
 * // Custom SnapCfg
 * SnapCfg cfg = new SnapCfg().setMaxForwardBytes(512).setPreferParagraph(false);
 * try (Chunker c = new Chunker("bert-base-uncased", 128, 128, cfg)) { ... }
 * }</pre>
 */
public final class Chunker implements AutoCloseable {

    static { NativeLoader.load(); }

    private long nativeHandle = 0;

    /**
     * Create a new Chunker with default {@link SnapCfg}.
     *
     * @param tokenizerRepoOrPath name of the HuggingFace repo (e.g., {@code "bert-base-uncased"})
     *                            or absolute path to a folder containing {@code tokenizer.json}.
     * @param chunkSize           target chunk size in tokens.
     * @param stride              number of tokens to overlap between chunks.
     * @throws IllegalArgumentException if any argument is invalid.
     * @throws RuntimeException         if native initialization fails.
     */
    public Chunker(String tokenizerRepoOrPath, int chunkSize, int stride) {
        this(tokenizerRepoOrPath, chunkSize, stride, new SnapCfg());
    }

    /**
     * Create a new Chunker with custom {@link SnapCfg}.
     *
     * @param tokenizerRepoOrPath name of the HuggingFace repo (e.g., {@code "bert-base-uncased"})
     *                            or absolute path to a folder containing {@code tokenizer.json}.
     * @param chunkSize           target chunk size in tokens.
     * @param stride              number of tokens to overlap between chunks.
     * @param snapCfg             configuration for sentence/paragraph boundary-snapping.
     * @throws IllegalArgumentException if any argument is invalid or {@code snapCfg} is null.
     * @throws RuntimeException         if native initialization fails.
     */
    public Chunker(String tokenizerRepoOrPath, int chunkSize, int stride, SnapCfg snapCfg) {
        if (chunkSize <= 0) throw new IllegalArgumentException("chunkSize must be > 0");
        if (stride    <= 0) throw new IllegalArgumentException("stride must be > 0");
        if (snapCfg   == null) throw new IllegalArgumentException("snapCfg must not be null");
        nativeCreate(tokenizerRepoOrPath, chunkSize, stride,
                snapCfg.getMaxForwardBytes(),
                snapCfg.getMaxBackwardBytes(),
                snapCfg.isPreferParagraph(),
                snapCfg.isTrimWhitespace());
        if (nativeHandle == 0)
            throw new RuntimeException("nativeCreate did not initialise the handle");
    }

    /**
     * Chunk a single text into byte-offset pairs.
     *
     * @param text input string, must not be null.
     * @return flat {@code int[]} of {@code [start0, end0, start1, end1, ...]}.
     *         Length is {@code 2 * numberOfChunks}. Empty array if text is empty
     *         or produces no tokens.
     */
    public int[] chunk(String text) {
        if (nativeHandle == 0) throw new IllegalStateException("Chunker already closed");
        return nativeChunk(text);
    }

    /**
     * Convenience: chunk and reconstruct all chunk strings in one call.
     * Calls {@link #chunk} then does the {@code getBytes}/{@code new String}
     * loop internally. Use this when you don't need the raw offsets.
     *
     * @param text input string, must not be null.
     * @return array of chunk strings, may be empty.
     */
    public String[] chunkToStrings(String text) {
        int[]  offsets = chunk(text);
        byte[] bytes   = text.getBytes(StandardCharsets.UTF_8);
        int    n       = offsets.length / 2;
        String[] out   = new String[n];
        for (int i = 0; i < n; i++) {
            int start = offsets[i * 2];
            int end   = offsets[i * 2 + 1];
            out[i] = new String(bytes, start, end - start, StandardCharsets.UTF_8);
        }
        return out;
    }

    @Override
    public void close() {
        if (nativeHandle != 0) { nativeDestroy(); nativeHandle = 0; }
    }

    private native void  nativeCreate(String tokenizerRepoOrPath,
                                      int chunkSize, int stride,
                                      int maxForwardBytes, int maxBackwardBytes,
                                      boolean preferParagraph, boolean trimWhitespace);
    private native void  nativeDestroy();
    private native int[] nativeChunk(String text);
}
