package com.model2vec;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
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
 * <p>FFM is used for native access to the Rust-backed chunker.</p>
 */
public final class Chunker implements AutoCloseable {

    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup LOOKUP = NativeLoader.load();

    private static final MethodHandle CREATE_MH = LINKER.downcallHandle(
            LOOKUP.find("chunker_create").get(),
            FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_BOOLEAN, ValueLayout.JAVA_BOOLEAN)
    );
    private static final MethodHandle FREE_MH = LINKER.downcallHandle(
            LOOKUP.find("chunker_free").get(),
            FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)
    );
    private static final MethodHandle CHUNK_MH = LINKER.downcallHandle(
            LOOKUP.find("chunker_chunk").get(),
            FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS)
    );
    private static final MethodHandle FREE_OFFSETS_MH = LINKER.downcallHandle(
            LOOKUP.find("chunker_free_offsets").get(),
            FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.JAVA_LONG)
    );

    private MemorySegment nativeHandle;

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
        
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment path = arena.allocateFrom(tokenizerRepoOrPath);
            this.nativeHandle = (MemorySegment) CREATE_MH.invokeExact(path, chunkSize, stride,
                    snapCfg.getMaxForwardBytes(), snapCfg.getMaxBackwardBytes(),
                    snapCfg.isPreferParagraph(), snapCfg.isTrimWhitespace());
        } catch (Throwable t) {
            throw new RuntimeException("chunker_create failed", t);
        }
        
        if (nativeHandle.equals(MemorySegment.NULL))
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
        if (nativeHandle.equals(MemorySegment.NULL)) throw new IllegalStateException("Chunker already closed");
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment textSeg = arena.allocateFrom(text);
            MemorySegment outLenSeg = arena.allocate(ValueLayout.JAVA_LONG);
            MemorySegment ptr = (MemorySegment) CHUNK_MH.invokeExact(nativeHandle, textSeg, outLenSeg);
            
            if (ptr.equals(MemorySegment.NULL)) return new int[0];
            
            long len = outLenSeg.get(ValueLayout.JAVA_LONG, 0);
            int[] offsets = new int[(int) len];
            MemorySegment.copy(ptr.reinterpret(len * 4), ValueLayout.JAVA_INT, 0, offsets, 0, (int) len);
            
            FREE_OFFSETS_MH.invokeExact(ptr, len);
            return offsets;
        } catch (Throwable t) {
            throw new RuntimeException("chunk failed", t);
        }
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
        if (!nativeHandle.equals(MemorySegment.NULL)) {
            try {
                FREE_MH.invokeExact(nativeHandle);
            } catch (Throwable t) {
                // ignore
            }
            nativeHandle = MemorySegment.NULL;
        }
    }
}
