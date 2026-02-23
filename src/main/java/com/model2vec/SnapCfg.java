package com.model2vec;

/**
 * Sentence/paragraph boundary-snapping configuration for {@link Chunker}.
 *
 * <pre>{@code
 * SnapCfg cfg = new SnapCfg()
 *     .setMaxForwardBytes(512)
 *     .setPreferParagraph(false);
 *
 * try (Chunker c = new Chunker("bert-base-uncased", 128, 64, cfg)) { ... }
 * }</pre>
 */
public final class SnapCfg {

    private int     maxForwardBytes  = 256;
    private int     maxBackwardBytes = 128;
    private boolean preferParagraph  = true;
    private boolean trimWhitespace   = true;

    /** Default: forward=256, backward=128, paragraph=true, trim=true. */
    public SnapCfg() {}

    public int     getMaxForwardBytes()    { return maxForwardBytes; }
    public int     getMaxBackwardBytes()   { return maxBackwardBytes; }
    public boolean isPreferParagraph()     { return preferParagraph; }
    public boolean isTrimWhitespace()      { return trimWhitespace; }

    public SnapCfg setMaxForwardBytes(int v) {
        if (v < 0) throw new IllegalArgumentException("maxForwardBytes must be >= 0");
        maxForwardBytes = v; return this;
    }
    public SnapCfg setMaxBackwardBytes(int v) {
        if (v < 0) throw new IllegalArgumentException("maxBackwardBytes must be >= 0");
        maxBackwardBytes = v; return this;
    }
    public SnapCfg setPreferParagraph(boolean v) { preferParagraph = v; return this; }
    public SnapCfg setTrimWhitespace(boolean v)  { trimWhitespace  = v; return this; }

    @Override
    public String toString() {
        return "SnapCfg{maxForwardBytes=" + maxForwardBytes +
               ", maxBackwardBytes=" + maxBackwardBytes +
               ", preferParagraph=" + preferParagraph +
               ", trimWhitespace=" + trimWhitespace + '}';
    }
}
