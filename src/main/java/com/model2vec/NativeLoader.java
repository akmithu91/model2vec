package com.model2vec;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/**
 * Extracts {@code libmodel2vec_jni.so} from the JAR at startup and loads it.
 * Linux x86_64 only.
 */
final class NativeLoader {

    private static volatile boolean loaded = false;

    private NativeLoader() {}

    static synchronized void load() {
        if (loaded) return;
        final String resource = "/native/libmodel2vec_jni.so";
        try (InputStream in = NativeLoader.class.getResourceAsStream(resource)) {
            if (in == null)
                throw new UnsatisfiedLinkError(
                    "Native library not found in JAR at " + resource +
                    " — run 'mvn package' to build it.");
            Path tmp = Files.createTempFile("model2vec_jni_", ".so");
            tmp.toFile().deleteOnExit();
            Files.copy(in, tmp, StandardCopyOption.REPLACE_EXISTING);
            System.load(tmp.toAbsolutePath().toString());
            loaded = true;
        } catch (IOException e) {
            throw new UnsatisfiedLinkError("Failed to extract native library: " + e.getMessage());
        }
    }
}
