package com.model2vec;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class Model2VecTest {

    private static final String MODEL =
        System.getProperty("model2vec.testModel", "minishlab/potion-base-8M");

    @Test void shapeMatchesSentenceCount() {
        try (Model2Vec m = new Model2Vec(MODEL, true, 128, 1024)) {
            float[][] e = m.encode(new String[]{"Hello world", "Rust is fast"});
            assertEquals(2, e.length);
            assertTrue(e[0].length > 0);
            assertEquals(e[0].length, e[1].length);
        }
    }

    @Test void normalizedEmbeddingHasUnitNorm() {
        try (Model2Vec m = new Model2Vec(MODEL, true, 512, 1024)) {
            float[] e = m.encode(new String[]{"Hello"})[0];
            double norm = 0;
            for (float v : e) norm += (double) v * v;
            assertEquals(1.0, Math.sqrt(norm), 1e-5);
        }
    }

    @Test void cosineSimilaritySelf() {
        try (Model2Vec m = new Model2Vec(MODEL, true, 512, 1024)) {
            float[] e = m.encode(new String[]{"Hello"})[0];
            assertEquals(1.0f, Model2Vec.cosineSimilarity(e, e), 1e-5f);
        }
    }

    @Test void twoIndependentInstances() {
        try (Model2Vec a = new Model2Vec(MODEL, true,  128, 1024);
             Model2Vec b = new Model2Vec(MODEL, false, 256, 512)) {
            assertEquals(a.encode(new String[]{"x"})[0].length,
                         b.encode(new String[]{"x"})[0].length);
        }
    }

    @Test void closedInstanceThrows() {
        Model2Vec m = new Model2Vec(MODEL, true, 128, 1024);
        m.close();
        assertThrows(IllegalStateException.class, () -> m.encode(new String[]{"hi"}));
    }

    @Test void concurrentEncodeNoException() throws Exception {
        try (Model2Vec m = new Model2Vec(MODEL, true, 128, 1024)) {
            var pool    = java.util.concurrent.Executors.newFixedThreadPool(8);
            var futures = new java.util.ArrayList<java.util.concurrent.Future<?>>();
            for (int t = 0; t < 8; t++)
                futures.add(pool.submit(() -> m.encode(new String[]{"concurrent test"})));
            for (var f : futures) f.get();
            pool.shutdown();
        }
    }
}
