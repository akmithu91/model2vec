package com.model2vec;

import org.junit.jupiter.api.Test;
import java.nio.charset.StandardCharsets;
import static org.junit.jupiter.api.Assertions.*;

class ChunkerTest {

    private static final String TOKENIZER =
            System.getProperty("chunker.testTokenizer", "bert-base-uncased");

    // Helper: reconstruct chunks from raw offsets
    private static String[] reconstruct(String text, int[] offsets) {
        byte[]   bytes = text.getBytes(StandardCharsets.UTF_8);
        String[] out   = new String[offsets.length / 2];
        for (int i = 0; i < out.length; i++) {
            int s = offsets[i * 2];
            int e = offsets[i * 2 + 1];
            out[i] = new String(bytes, s, e - s, StandardCharsets.UTF_8);
        }
        return out;
    }

    @Test void shortTextYieldsOneChunk() {
        try (Chunker c = new Chunker(TOKENIZER, 512, 256)) {
            int[] off = c.chunk("Short text.");
            assertEquals(2, off.length, "one chunk = two ints");
        }
    }

    @Test void emptyTextYieldsEmptyArray() {
        try (Chunker c = new Chunker(TOKENIZER, 128, 64)) {
            assertEquals(0, c.chunk("").length);
        }
    }

    @Test void longTextProducesMultipleChunks() {
        String text = "This is a sentence. ".repeat(200);
        try (Chunker c = new Chunker(TOKENIZER, 64, 64)) {
            int[] off = c.chunk(text);
            assertTrue(off.length > 2, "expected multiple chunks");
        }
    }

    @Test void offsetsAreEvenLength() {
        try (Chunker c = new Chunker(TOKENIZER, 128, 64)) {
            int[] off = c.chunk("Hello world. How are you?");
            assertEquals(0, off.length % 2, "must be pairs");
        }
    }

    @Test void offsetsAreValid() {
        String text = "One. Two. Three. Four. Five.";
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        try (Chunker c = new Chunker(TOKENIZER, 8, 8)) {
            int[] off = c.chunk(text);
            assertTrue(off.length % 2 == 0, "must be pairs");
            for (int i = 0; i < off.length; i += 2) {
                int s = off[i], e = off[i + 1];
                assertTrue(s >= 0,             "start >= 0");
                assertTrue(e <= bytes.length,  "end <= text byte length");
                assertTrue(s < e,              "start < end (non-empty chunk)");
            }
            // Note: end[i] > start[i+1] is valid — sentence snapping can push
            // a chunk's end forward past the next chunk's token start boundary.
        }
    }

    @Test void reconstructedChunksAreNonBlank() {
        String   text = "Sentence one. Sentence two! Sentence three? Yes.";
        try (Chunker c = new Chunker(TOKENIZER, 8, 8)) {
            String[] chunks = reconstruct(text, c.chunk(text));
            for (String chunk : chunks)
                assertFalse(chunk.isBlank(), "chunk must not be blank");
        }
    }

    @Test void chunkToStringsMatchesManualReconstruct() {
        String text = "Financial markets rallied today. Tech stocks led gains.";
        try (Chunker c = new Chunker(TOKENIZER, 32, 32)) {
            int[]    offsets  = c.chunk(text);
            String[] manual   = reconstruct(text, offsets);
            String[] helper   = c.chunkToStrings(text);
            assertArrayEquals(manual, helper);
        }
    }

    @Test void customSnapCfg() {
        SnapCfg cfg = new SnapCfg()
                .setMaxForwardBytes(512)
                .setMaxBackwardBytes(64)
                .setPreferParagraph(false);
        try (Chunker c = new Chunker(TOKENIZER, 128, 128, cfg)) {
            int[] off = c.chunk("Hello world.");
            assertEquals(0, off.length % 2);
        }
    }

    @Test void closedChunkerThrows() {
        Chunker c = new Chunker(TOKENIZER, 128, 64);
        c.close();
        assertThrows(IllegalStateException.class, () -> c.chunk("hi"));
    }

    @Test void invalidChunkSizeThrows() {
        assertThrows(IllegalArgumentException.class,
                () -> new Chunker(TOKENIZER, 0, 64));
    }

    @Test void invalidStrideThrows() {
        assertThrows(IllegalArgumentException.class,
                () -> new Chunker(TOKENIZER, 128, 0));
    }
}