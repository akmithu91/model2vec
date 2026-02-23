package com.model2vec;

/**
 * An instance of a loaded Model2Vec embedding model.
 *
 * <p>Each instance owns one native {@code Arc<StaticModel>} stored as a raw
 * pointer in {@code nativeHandle}. Multiple instances may coexist with
 * different models or different encode parameters.</p>
 *
 * <h3>Threading</h3>
 * <p>{@link #encode} is fully thread-safe with zero locking. The Rust model
 * is {@code Sync} — multiple threads may call encode on the same instance
 * simultaneously.</p>
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

    static { NativeLoader.load(); }

    private long nativeHandle = 0;

    /**
     * Load a Model2Vec model.
     *
     * @param modelId    HuggingFace repo ID or absolute local directory path.
     * @param normalize  {@code true} to L2-normalise embeddings.
     * @param maxLength  Max tokens per sentence. Pass {@code <= 0} for model
     *                   default (512). Should match your {@link Chunker} chunk size.
     * @param batchSize  Internal batch size for encoding. Pass {@code <= 0} for
     *                   default (1024). Tune down only if hitting memory pressure.
     */
    public Model2Vec(String modelId, boolean normalize, int maxLength, int batchSize) {
        nativeCreate(modelId, normalize, maxLength, batchSize);
        if (nativeHandle == 0)
            throw new RuntimeException("nativeCreate did not initialise the handle");
    }

    /**
     * Encode a batch of sentences into embedding vectors.
     *
     * @param sentences non-null array of non-null strings.
     * @return {@code float[sentences.length][embeddingDim]}
     */
    public float[][] encode(String[] sentences) {
        if (nativeHandle == 0) throw new IllegalStateException("Model already closed");
        return nativeEncode(sentences);
    }

    @Override
    public void close() {
        if (nativeHandle != 0) { nativeDestroy(); nativeHandle = 0; }
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

    private native void      nativeCreate(String modelId, boolean normalize,
                                          int maxLength, int batchSize);
    private native void      nativeDestroy();
    private native float[][] nativeEncode(String[] sentences);
}
