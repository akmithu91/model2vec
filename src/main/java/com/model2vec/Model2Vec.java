package com.model2vec;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * An instance of a loaded Model2Vec embedding model.
 *
 * <p>Each instance owns one native {@code Arc<StaticModel>} stored as a raw
 * pointer in {@code nativeHandle}. Multiple instances may coexist with
 * different models or different encode parameters.</p>
 *
 * <p>FFM is used for native access, and a memory pool of native segments
 * is used to minimize allocations and maximize performance.</p>
 *
 * <pre>{@code
 * // maxLength <= 0 means use model default (512)
 * // batchSize <= 0 means use default (1024)
 * try (Model2Vec model = new Model2Vec("minishlab/potion-base-8M", true, 128, 1024)) {
 *     float[][] embeddings = model.encode(new String[]{"Hello", "World"});
 * }
 * }</pre>
 */
public final class Model2Vec implements AutoCloseable {

    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup LOOKUP = NativeLoader.load();

    private static final MethodHandle CREATE_MH = LINKER.downcallHandle(
            LOOKUP.find("model2vec_create").get(),
            FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_BOOLEAN, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT)
    );
    private static final MethodHandle FREE_MH = LINKER.downcallHandle(
            LOOKUP.find("model2vec_free").get(),
            FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)
    );
    private static final MethodHandle GET_DIM_MH = LINKER.downcallHandle(
            LOOKUP.find("model2vec_get_dim").get(),
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)
    );
    private static final MethodHandle ENCODE_MH = LINKER.downcallHandle(
            LOOKUP.find("model2vec_encode").get(),
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS)
    );

    private MemorySegment nativeHandle;
    private final int dim;
    private final int maxBatchSize;
    private final Arena poolArena = Arena.ofShared();
    private final ConcurrentLinkedQueue<MemorySegment> segmentPool = new ConcurrentLinkedQueue<>();

    /**
     * Load a Model2Vec model.
     *
     * @param modelId    HuggingFace repo ID or absolute local directory path.
     * @param normalize  {@code true} to L2-normalise embeddings.
     * @param maxLength  Max tokens per sentence. Pass {@code <= 0} for model
     *                   default (512). Should match your {@link Chunker} chunk size.
     * @param batchSize  Internal batch size for encoding. Pass {@code <= 0} for
     *                   default (1024).
     */
    public Model2Vec(String modelId, boolean normalize, int maxLength, int batchSize) {
        this.maxBatchSize = batchSize > 0 ? batchSize : 1024;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment id = arena.allocateFrom(modelId);
            this.nativeHandle = (MemorySegment) CREATE_MH.invokeExact(id, normalize, maxLength, batchSize);
        } catch (Throwable t) {
            throw new RuntimeException("nativeCreate failed", t);
        }
        if (nativeHandle.equals(MemorySegment.NULL))
            throw new RuntimeException("nativeCreate did not initialise the handle");

        try {
            this.dim = (int) GET_DIM_MH.invokeExact(nativeHandle);
        } catch (Throwable t) {
            throw new RuntimeException("getDim failed", t);
        }
    }

    /**
     * Encode a batch of sentences into embedding vectors.
     *
     * @param sentences non-null array of non-null strings.
     * @return {@code float[sentences.length][embeddingDim]}
     */
    public float[][] encode(String[] sentences) {
        if (nativeHandle.equals(MemorySegment.NULL)) throw new IllegalStateException("Model already closed");
        float[][] out = new float[sentences.length][dim];
        
        // We use a contiguous native buffer for Rust to fill, then copy to float[][].
        // This minimizes JNI/FFM boundary crossings and allows using a memory pool.
        MemorySegment buffer = segmentPool.poll();
        long requiredSize = (long) sentences.length * dim * 4;
        if (buffer == null || buffer.byteSize() < requiredSize) {
            // Allocate a buffer large enough for a full batch if we need a new one
            long allocSize = Math.max(requiredSize, (long) maxBatchSize * dim * 4);
            buffer = poolArena.allocate(allocSize, 4);
        }

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment sentencesPtrs = arena.allocate(ValueLayout.ADDRESS, sentences.length);
            for (int i = 0; i < sentences.length; i++) {
                sentencesPtrs.setAtIndex(ValueLayout.ADDRESS, i, arena.allocateFrom(sentences[i]));
            }

            int res = (int) ENCODE_MH.invokeExact(nativeHandle, sentencesPtrs, (long) sentences.length, buffer);
            if (res != 0) throw new RuntimeException("encode failed with code " + res);

            // Copy results to float[][]
            for (int i = 0; i < sentences.length; i++) {
                MemorySegment.copy(buffer, ValueLayout.JAVA_FLOAT, (long) i * dim * 4, out[i], 0, dim);
            }
        } catch (Throwable t) {
            throw new RuntimeException("encode failed", t);
        } finally {
            segmentPool.offer(buffer);
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
            poolArena.close();
        }
    }

    /**
     * Cosine similarity between two embedding vectors.
     * If {@code normalize=true} was used this is a plain dot product.
     */
    public static float cosineSimilarity(float[] a, float[] b) {
        if (a.length != b.length)
            throw new IllegalArgumentException("length mismatch: " + a.length + " vs " + b.length);
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += (double) a[i] * b[i];
            na  += (double) a[i] * a[i];
            nb  += (double) b[i] * b[i];
        }
        return (na == 0 || nb == 0) ? 0f : (float)(dot / (Math.sqrt(na) * Math.sqrt(nb)));
    }
}
